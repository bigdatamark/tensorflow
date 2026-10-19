# [MODULE DOC]
# AutoGraph control flow operators: functional forms of for, while and if statements, plus the
# verification helpers that check loop and conditional variables for consistent types, shapes and
# structure. Each operator picks between a staged TensorFlow implementation (while loops, cond,
# datasets, iterators, ranges) and a plain Python implementation depending on the runtime types of
# its inputs. State mutated by the code blocks is captured with paired get-state / set-state
# callbacks. Also exposes module-level tuning constants (maximum Python loop iterations and
# thresholds for the inefficient-unroll warning) and a registry that lets other types provide their
# own for-loop implementation. Imported by the converted code through the operators package.
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
"""Control flow statements: loops, conditionals, etc.

Note: most of these operators accept pairs of get_state/set_state functions, to
capture mutations that the corresponding code blocks might make. These
mutations only need to be captured when staging the control flow, and they just
work when reverting to Python behavior.

__Examples__

```
while cond:
  self.x += i
```

When the functionalized version is executed as a Python loop, it just works:

```
def loop_body():
  self.x += i     # works as expected for Python loops
```

But it won't work for TF loops:

```
def loop_body():
  self.x += i     # self.x has the wrong value!
```

get_state/set_state allow piping the mutations through the loop variables as
well, in effect changing the loop body:

```
def loop_body(self_x):
  self.x = self_x  # self.x now has the proper value
  self.x += i      # the original block
  self_x = self.x  # write self.x back into the loop vars
  return self_x

self_x = tf.while_loop(...)
self.x = self_x    # the result is not properly captured
```
"""

import functools
import sys
import traceback

import numpy as np

from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils


PYTHON_MAX_ITERATIONS = 100000000  # Fails in about one minute for empty loops.
WARN_INEFFICIENT_UNROLL = True
INEFFICIENT_UNROLL_MIN_ITERATIONS = 50000
INEFFICIENT_UNROLL_MIN_OPS = 1


# TODO(mdan): Use the custom operator pattern instead of type dispatch.
# An example of this pattern is found in the implementation of distributed
# datasets. Before it can be used though, we need to standardize the interface.

for_loop_registry = type_registry.TypeRegistry()


def _is_none_or_undef(value):
  # [FUNCTION DOC]
  # HELPER: predicate used when validating loop and conditional variables.
  # Returns true if value is the null value or one of the AutoGraph markers for an undefined symbol
  # or an undefined return value; false otherwise.
  """Tests whether a value is None or undefined.

  AutoGraph represents undefined symbols using special objects of type Undefined
  or UndefinedReturnValue.

  Args:
    value: value to test

  Returns:
    Boolean
  """
  return ((value is None)
          or isinstance(value, variables.UndefinedReturnValue)
          or isinstance(value, variables.Undefined))


def _verify_tf_condition(cond, tag):
  # [FUNCTION DOC]
  # HELPER: validates and normalizes a condition used by a staged conditional or loop.
  # Inputs: cond (anything convertible to a tensor), tag (text naming the construct, used in error
  # messages). Converts to a tensor; raises a value error if the type is not boolean, or if the
  # known dimensions of a non-scalar shape multiply to more than one element. Unknown-rank shapes
  # and single-element shapes are reshaped to a scalar. Returns the scalar boolean tensor.
  """Ensures that the condition can be used in a TF control flow."""
  extra_hint = 'to check for None, use `is not None`'
  cond = tensor_conversion.convert_to_tensor_v2(cond)

  if cond.dtype != dtypes.bool:
    raise ValueError(
        'condition of {} expected to be `tf.bool` scalar, got {}'
        '; to use as boolean Tensor, use `tf.cast`'
        '; {}'.format(tag, cond, extra_hint))

  if cond.shape is None or cond.shape.ndims is None:
    # TODO(mdan): Consider a explicit size check, if not too slow.
    cond = array_ops.reshape(cond, ())

  elif cond.shape.ndims > 0:
    known_dims = [d for d in cond.shape.as_list() if d is not None]
    if np.prod(known_dims) > 1:
      raise ValueError(
          'condition of {} expected to be `tf.bool` scalar, got {}'
          '; {}'.format(tag, cond, extra_hint))
    else:
      cond = array_ops.reshape(cond, ())

  return cond


def verify_loop_init_vars(
    init_vars, symbol_names, first_iter_vars=None, extra_message=None
):
  # [FUNCTION DOC]
  # Checks that the loop variables captured before a staged loop are usable.
  # Inputs: init_vars (initial values), symbol_names (matching names, same length), first_iter_vars
  # (optional values after one iteration, used only to word an error; defaults to all null),
  # extra_message (optional text appended to the "must be defined" error).
  # Returns nothing; does nothing when there are no names. Raises a value error for an undefined
  # return value whose first-iteration value is truthy, a not-implemented error for an undefined
  # return value otherwise (return inside the loop), and a value error for a variable that is the
  # null value or undefined before the loop. Length mismatches trip assertions.
  """Ensures that all values in the state are valid to use in a TF loop.

  The init_vars may contain placeholder values derived from first_iter_vars.

  Args:
    init_vars: initial loop variables (as taken before entering the loop)
    symbol_names: corresponding names of the initial loop variables
    first_iter_vars: loop variables after one iteration of the loop
    extra_message: an extra string to append to the error message, in case of
      "undefined variable" errors (see variables.Undefined)
  """
  if not symbol_names:
    return
  if first_iter_vars is None:
    first_iter_vars = (None,) * len(symbol_names)

  assert len(symbol_names) == len(init_vars)
  assert len(symbol_names) == len(first_iter_vars)
  for name, val, fi_val in zip(symbol_names, init_vars, first_iter_vars):
    if isinstance(val, variables.UndefinedReturnValue):
      if fi_val:
        raise ValueError(
            'the return value from a TensorFlow loop may only be a {}; got {}'
            .format(LEGAL_LOOP_TYPES, type(fi_val)))
      else:
        # TODO(mdan): This can be handled by removing the return value.
        raise NotImplementedError(
            'a return statement cannot be placed inside this TensorFlow loop;'
            ' this may happen if a return statement depends on a'
            ' static Python condition such as a hyperparameter')

    error_msg = None
    if val is None:
      error_msg = "'{}' is not allowed to be None before the loop".format(name)
    elif isinstance(val, variables.Undefined):
      error_msg = "'{}' must be defined before the loop".format(name)
      if extra_message:
        error_msg += '\n' + extra_message

    if error_msg is not None:
      raise ValueError(error_msg)


def _is_subshape(left, right):
  # [FUNCTION DOC]
  # HELPER: shape comparison for loop variable checks (an asymmetric version of compatibility).
  # Returns true if the left shape is at least as specific as the right shape: always true when the
  # right shape has unknown rank, false when ranks differ, and otherwise true when every dimension
  # known on the right equals the left dimension.
  """Returns True if left shape is at least as specific as right shape."""
  # TODO(mdan): This code should be in TensorShape.
  # Note: this is not the same as TensorShape.is_compatible_with, which is
  # symmetric.
  # This code also duplicates _ShapeLessThanOrEqual from  control_flow_ops.py.
  if right.dims is None:
    return True
  if left.ndims != right.ndims:
    return False
  for ldim, rdim in zip(left.dims, right.dims):
    if rdim.value is not None and ldim.value != rdim.value:
      return False
  return True


# TODO(mdan): Remove these verifications once TF ops can properly report names.
def _verify_single_loop_var(
    name, check_shape, init, entry, exit_, shape_invariant):
  # [FUNCTION DOC]
  # HELPER: checks one loop variable for consistency across the loop.
  # Inputs: name (for messages), check_shape (whether to compare shapes), init / entry / exit_
  # (values before the loop, at iteration entry, and at iteration exit), shape_invariant (optional
  # declared shape the variable must satisfy). Python scalars, strings and arrays are converted to
  # tensors first. Returns nothing; silently returns if entry or exit is not tensor-like or lacks
  # dtype/shape. Raises a value error if exit is the null value, a type error if the dtypes of
  # entry and exit differ, and a value error if shapes change without an invariant or violate the
  # given invariant. An assertion fails if entry is the null value.
  """Verifies whether the initial, entry and exit values are consistent."""
  assert entry is not None, "no TF op should set '{}' to None?".format(name)
  if exit_ is None:
    raise ValueError("'{}' is None at the end of the iteration.".format(name))

  if isinstance(init, (bool, int, float, str, np.ndarray)):
    init = tensor_conversion.convert_to_tensor_v2(init)
  if isinstance(entry, (bool, int, float, str, np.ndarray)):
    entry = tensor_conversion.convert_to_tensor_v2(entry)
  if isinstance(exit_, (bool, int, float, str, np.ndarray)):
    exit_ = tensor_conversion.convert_to_tensor_v2(exit_)

  if (not tensor_util.is_tf_type(entry) or
      not tensor_util.is_tf_type(exit_)):
    return

  # TODO(mdan): Properly account for CompositeTensors.
  if (not hasattr(entry, 'dtype') or
      not hasattr(exit_, 'dtype')):
    return
  if (not hasattr(entry, 'shape') or
      not hasattr(exit_, 'shape')):
    return

  if entry.dtype != exit_.dtype:
    raise TypeError(
        "'{}' has dtype {} before the loop, but dtype {} after one"
        ' iteration'.format(
            name,
            entry.dtype.name,
            exit_.dtype.name,
        ))
  if check_shape:
    exit_shape = exit_.shape
    if shape_invariant is None:
      entry_shape = entry.shape
      if not _is_subshape(exit_shape, entry_shape):
        raise ValueError(
            "'{}' has shape {} before the loop, but shape {} after one"
            ' iteration. Use tf.autograph.experimental.set_loop_options to set'
            ' shape invariants.'.format(name, entry_shape, exit_shape))
    else:
      init_shape = init.shape
      if not _is_subshape(init_shape, shape_invariant):
        raise ValueError(
            "'{}' has shape {} before the loop, which does not conform with"
            ' the shape invariant {}.'.format(name, init_shape,
                                              shape_invariant))
      if not _is_subshape(exit_shape, shape_invariant):
        raise ValueError(
            "'{}' has shape {} after one iteration, which does not conform with"
            ' the shape invariant {}.'.format(name, exit_shape, shape_invariant)
        )


def verify_tf_loop_vars(
    init_vars,
    iter_entry_vars,
    iter_exit_vars,
    symbol_names,
    opts,
    check_shapes=True,
):
  # [FUNCTION DOC]
  # Verifies all loop variables for a staged loop.
  # Inputs: init_vars, iter_entry_vars, iter_exit_vars (sequences of values, one per name),
  # symbol_names (names, same length), opts (mapping of loop options; may hold shape invariants),
  # check_shapes (default true; when false shape invariants are ignored and shape checks skipped).
  # For each variable, checks that the nested structure matches between initial and entry values
  # (retrying with variables converted to tensors), between entry and exit values, and between
  # initial value and any shape invariant, then checks dtypes and shapes. Returns nothing; raises
  # type errors for structure mismatches (chained to the original cause) and the errors of the
  # single-variable check. Length mismatches trip assertions.
  """Verifies loop variables for consistency."""
  if check_shapes and 'shape_invariants' in opts:
    shape_invariants = opts['shape_invariants']
  else:
    shape_invariants = nest.map_structure(lambda _: None, iter_entry_vars)

  assert len(symbol_names) == len(shape_invariants)
  assert len(symbol_names) == len(init_vars)
  assert len(symbol_names) == len(iter_entry_vars)
  assert len(symbol_names) == len(iter_exit_vars)

  for i in range(len(symbol_names)):
    name = symbol_names[i]
    init = init_vars[i]
    entry = iter_entry_vars[i]
    exit_ = iter_exit_vars[i]
    invariant = shape_invariants[i]

    try:
      nest.assert_same_structure(init, entry, expand_composites=True)
    except (ValueError, TypeError):
      # `Variable`s in `init` may be implicitly converted to `Tensor`s. Convert
      # `ResourceVariable`s to Tensors so tf.nest.assert_same_structure
      # won't break due to type spec mismatches between `ResourceVariable`s and
      # `Tensor`s.
      try:
        init_tensors = variable_utils.convert_variables_to_tensors(init)
        nest.assert_same_structure(init_tensors, entry, expand_composites=True)
      except (ValueError, TypeError) as e:
        raise TypeError("'{}' does not have the same nested structure after one"
                        ' iteration.\n\n{}'.format(name, e)) from e

    try:
      nest.assert_same_structure(entry, exit_, expand_composites=True)
    except (ValueError, TypeError) as e:
      raise TypeError("'{}' does not have the same nested structure after one"
                      ' iteration.\n\n{}'.format(name, e)) from e
    if invariant is not None:
      try:
        nest.assert_same_structure(init, invariant, expand_composites=False)
      except (ValueError, TypeError) as e:
        raise TypeError("'{}' does not have the same nested structure as its"
                        ' corresponding shape invariant.\n\n{}'.format(
                            name, e)) from e

    nest.map_structure(
        functools.partial(_verify_single_loop_var, name, check_shapes), init,
        entry, exit_, invariant)


def verify_single_cond_var(name, body_var, orelse_var):
  # [FUNCTION DOC]
  # Checks one variable's values from the two branches of a conditional.
  # Inputs: name (for messages), body_var and orelse_var (values at the end of the main and else
  # branches). Python scalars, strings and arrays are converted to tensors. Raises a value error
  # if either value is the null value, and a type error if both are tensors with different
  # dtypes. Returns nothing; non-tensor values are accepted silently.
  """Verifies whether body_var and orelse_var are consistent."""
  if body_var is None:
    raise ValueError("'{}' is None at the end of the main branch.".format(name))
  if orelse_var is None:
    raise ValueError(
        "'{}' is None at the end of the else branch.".format(name))

  if isinstance(body_var, (bool, int, float, str, np.ndarray)):
    body_var = tensor_conversion.convert_to_tensor_v2(body_var)

  if isinstance(orelse_var, (bool, int, float, str, np.ndarray)):
    orelse_var = tensor_conversion.convert_to_tensor_v2(orelse_var)

  if (not tensor_util.is_tf_type(body_var) or
      not tensor_util.is_tf_type(orelse_var)):
    return

  # TODO(mdan): Properly account for CompositeTensors.
  if (not hasattr(body_var, 'dtype') or
      not hasattr(orelse_var, 'dtype')):
    return

  if body_var.dtype != orelse_var.dtype:
    raise TypeError(
        "'{}' has dtype {} in the main branch, but dtype {} in the else"
        ' branch'.format(name, body_var.dtype.name,
                         orelse_var.dtype.name))


def _verify_tf_cond_branch_vars(vars_, symbol_names, branch_name):
  # [FUNCTION DOC]
  # HELPER: checks the output of one conditional branch. Inputs: vars_ (values produced), their
  # symbol_names, and branch_name (text used in messages). Raises a value error for any variable
  # left undefined, or for an undefined return value (the branch lacks a return). Returns nothing.
  """Verifies variables output by a conditional branch for consistency."""
  for name, var_ in zip(symbol_names, vars_):
    if isinstance(var_, variables.Undefined):
      raise ValueError(
          "'{}' must also be initialized in the {} branch".format(
              name, branch_name))
    if isinstance(var_, variables.UndefinedReturnValue):
      raise ValueError(
          'the {} branch must also have a return statement.'.format(
              branch_name))


def _verify_tf_cond_vars(body_vars, orelse_vars, symbol_names):
  # [FUNCTION DOC]
  # HELPER: checks that both branches of a conditional produce consistent variables.
  # Inputs: body_vars and orelse_vars (values per name from each branch) and symbol_names. For each
  # name, requires the same nested structure (retrying after converting variables to tensors) and
  # then runs the single-variable dtype check on every leaf. Raises a type error naming the
  # variable on structure mismatch; returns nothing.
  """Verifies variables manipulated by a conditional for consistency."""
  named_vars = zip(symbol_names, body_vars, orelse_vars)

  for name, body_var, orelse_var in named_vars:
    try:
      nest.assert_same_structure(body_var, orelse_var, expand_composites=True)
    except (ValueError, TypeError):
      # One branch of cond could be a `Tensor`, while the other branch could be
      # a `ResourceVariable`. Convert `ResourceVariable`s to `Tensor`s so
      # assert_same_structure won't fail.
      try:
        body_var_tensors = variable_utils.convert_variables_to_tensors(body_var)
        orelse_var_tensors = variable_utils.convert_variables_to_tensors(
            orelse_var)
        nest.assert_same_structure(body_var_tensors, orelse_var_tensors,
                                   expand_composites=True)
      except (ValueError, TypeError) as e:
        raise TypeError(
            "'{}' must have the same nested structure in the main and else"
            ' branches:\n\n{}'.format(name, str(e))) from e
    nest.map_structure(
        functools.partial(verify_single_cond_var, name), body_var, orelse_var)


def for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  # [FUNCTION DOC]
  # Functional form of a for statement; dispatches on the type of the iterated entity.
  # Inputs: iter_ (entity iterated), extra_test (callback returning a boolean extra loop condition,
  # or null), body (callback taking the loop target), get_state / set_state (callbacks to read and
  # restore the loop state), symbol_names (names of the state variables), opts (mapping of loop
  # options). Selection: a type registered in the for-loop registry wins; otherwise a tensor that
  # is a range uses the range version, other tensors the known-length version, distributed
  # iterators and distributed iterables their own versions, and everything else the Python loop.
  # Returns nothing; results flow through the state callbacks.
  """Functional form of a for statement.

  The loop operates on a state, which includes all symbols that are
  variant across loop iterations, excluding the variables local to the loop.

  For example, given the loop below that calculates the geometric and
  arithmetic means or some numbers:

  ```
    geo_mean = 1
    arith_mean = 0
    for i in range(n):
      a = numbers[i]
      geo_mean *= a
      arith_mean += a
  ```

  The state is represented by the variables named geo_mean and arith_mean. The
  `extra_test`, `body`, `get_state` and `set_state` functions must bind to the
  original `geo_mean` and `arith_mean` symbols, using `nonlocal`.

  The inputs and outputs of the callables representing the loop blocks are not
  explicit - instead, these functions must use nonlocal/global for side effects.
  The inputs and outputs are instead controlled by the set_state/get_state
  functions.

  Args:
    iter_: The entity being iterated over.
    extra_test: Callable with boolean return type. An additional loop condition.
    body: Callable representing the actual loop body.
    get_state: Additional callable which can capture additional state (such as
      the values of composite symbols). This is only useful when staging the
      loop.
    set_state: Additional callable which save values captured by get_state back
      into the Python environment. This is only useful when staging the loop.
    symbol_names: Tuple containing names of the loop variables returned by
      get_state.
    opts: Optional dict of extra loop parameters.
  """

  try:
    for_fn = for_loop_registry.lookup(iter_)
  except LookupError:
    for_fn = _py_for_stmt

    if tensor_util.is_tf_type(iter_):
      if tensors.is_range_tensor(iter_):
        for_fn = _tf_range_for_stmt
      else:
        for_fn = _known_len_tf_for_stmt
    elif isinstance(iter_, distribute.Iterator):
      for_fn = _tf_iterator_for_stmt
    elif isinstance(iter_, distribute.Iterable):
      # TODO(b/162250181): Use _tf_iterator_for_stmt(iter(iter_)...
      for_fn = _tf_distributed_iterable_for_stmt

  for_fn(iter_, extra_test, body, get_state, set_state, symbol_names, opts)


def _py_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts
):
  # [FUNCTION DOC]
  # HELPER: plain-Python implementation of the for statement operator.
  # Same inputs as the dispatcher; the state callbacks, names and options are ignored. When
  # running with debug assertions on, a loop checker is notified around every iteration (to warn
  # about inefficient unrolling or enforce the iteration limit). The extra test is evaluated before
  # each iteration (including before the first) and converted to a boolean; if it is a graph
  # tensor, the conversion error is turned into a not-implemented error explaining that break and
  # return depending on a tensor condition are unsupported in Python loops. Returns nothing.
  """Overload of for_stmt that executes a Python for loop."""
  del get_state, set_state, symbol_names, opts

  if __debug__:
    checker = _PythonLoopChecker()
    before_iteration = checker.before_iteration
    after_iteration = checker.after_iteration
    before_iteration()

    original_body = body
    def protected_body(protected_iter):
      # [FUNCTION DOC]
      # HELPER: wraps the loop body for the Python loop checker; runs the original body with the
      # loop target, then reports the end of the iteration and the start of the next. Returns
      # nothing.
      original_body(protected_iter)
      after_iteration()
      before_iteration()
    body = protected_body

  if extra_test is not None:
    def guarded_extra_test():
      # [FUNCTION DOC]
      # HELPER: evaluates the extra loop condition and converts it to a boolean. If conversion
      # fails because the value is a graph tensor, logs the error at verbosity 1 and raises a
      # not-implemented error (chained to the cause) pointing to the AutoGraph limitations page.
      extra_test_result = extra_test()
      try:
        # Note: Using try/except and not tensor_util.is_tf_type to avoid
        # performance degradation.
        return bool(extra_test_result)
      except errors_impl.OperatorNotAllowedInGraphError as e:
        ag_logging.log(
            1,
            'Caught error while evaluating loop stop condition',
            exc_info=True)
        # TODO(mdan): We can pass the location of extra_test and show it here.
        raise NotImplementedError(
            'break and return statements which depend on a TF condition are not'
            ' supported in Python for loops. Did you intend to make it a TF'
            ' loop?\nSee '
            'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/'
            'python/autograph/g3doc/reference/limitations.md'
            '#consistency-of-control-flow-types for more info.') from e

    if guarded_extra_test():
      for target in iter_:
        body(target)
        if not guarded_extra_test():
          break

  else:
    for target in iter_:
      body(target)


def _add_max_iterations_hint(opts, n):
  # [FUNCTION DOC]
  # HELPER: records a maximum iteration count for a staged loop. Inputs: opts (mutable mapping of
  # loop options) and n (iteration count, scalar or tensor). Only when the default graph (or a
  # parent) is inside an XLA context, sets the maximum iterations option to n; otherwise does
  # nothing. Mutates opts; returns nothing.
  # TODO(b/159186914): Remove the safeguard, and always set maximum_iterations.
  if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
    opts['maximum_iterations'] = n


def _known_len_tf_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  # [FUNCTION DOC]
  # HELPER: staged for loop over a tensor (or similar) with a known length.
  # Same inputs as the dispatcher. Unstacks the iterated tensor into a tensor array, then runs a
  # staged while loop whose extra state is an iteration index prepended to the user state. The
  # loop continues while the index is below the length and, if given, the extra test holds
  # (evaluated through a conditional). Adds the length as a maximum-iterations hint (XLA only).
  # Mutates opts; returns nothing.
  """Overload of for_stmt that iterates over TF entities that admit a length."""
  n = py_builtins.len_(iter_)

  # TODO(b/117628877): Revisit performance once XLA has the necessary support.
  # Note: using a TensorArray creates an extra copy, but can calculate
  # gradients more efficiently than StridedSlice.
  ta = tensor_array_ops.TensorArray(iter_.dtype, size=n)
  iter_ = ta.unstack(iter_)

  iterate_index = 0

  def aug_get_state():
    # [FUNCTION DOC]
    # HELPER: returns the user state with the iteration index prepended.
    return (iterate_index,) + get_state()

  def aug_set_state(aug_loop_vars):
    # [FUNCTION DOC]
    # HELPER: splits the augmented state into the iteration index (stored) and the user state
    # (passed to the user set-state callback). Returns nothing.
    nonlocal iterate_index
    # TODO(b/171479293): Drop the lint override.
    iterate_index, *loop_vars = aug_loop_vars  # pylint:disable=unused-variable
    # The iteration index is not "output" by the for loop. If the iteration index
    # is used outside the loop, it will appear in the loop vars separately.
    set_state(loop_vars)

  def aug_body():
    # [FUNCTION DOC]
    # HELPER: loop body; reads the element at the current index from the tensor array, runs the
    # user body on it, and increments the index. Returns nothing.
    nonlocal iterate_index
    body(iter_.read(iterate_index))
    iterate_index += 1

  def aug_test():
    # [FUNCTION DOC]
    # HELPER: loop condition; true while the index is below the length and, if present, the extra
    # test holds. Returns a boolean tensor.
    main_test = iterate_index < n
    if extra_test is not None:
      return tf_cond.cond(main_test, extra_test, lambda: False)
    return main_test

  _add_max_iterations_hint(opts, n)

  _tf_while_stmt(
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      ('<internal iterate>',) + symbol_names,
      opts,
  )


def _tf_range_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  # [FUNCTION DOC]
  # HELPER: staged for loop over a range tensor, without materializing the range.
  # Same inputs as the dispatcher; the options must contain the name of the loop target. Reads
  # start, limit and step from the range operation's inputs and runs a staged while loop with the
  # current value prepended to the user state. The user variable holding the loop target, if still
  # undefined, is initialized to the start value. The test handles positive and negative steps
  # (using the constant step when known, otherwise a tensor expression) and the optional extra
  # test. Adds the range length as a maximum-iterations hint (XLA only). Mutates opts; returns
  # nothing.
  """Overload of for_stmt that iterates over a TF range (and elides it)."""
  start, limit, delta = iter_.op.inputs

  iterate = start

  def _value_or(name, var, default):
    # [FUNCTION DOC]
    # HELPER: returns default if name is the loop target name and var is still undefined;
    # otherwise returns var.
    # KNOWN ISSUE: compares name with the whole option entry for iterate names, which elsewhere
    # looks like it may be a collection of names; a match then only works if it is a single name.
    if (name == opts['iterate_names'] and isinstance(var, variables.Undefined)):
      return default
    return var

  def aug_get_state():
    # [FUNCTION DOC]
    # HELPER: returns the current iterate followed by the user state, with an undefined loop
    # target replaced by the current iterate.
    state_vars = get_state()
    state_vars = tuple(
        _value_or(name, var, iterate)
        for name, var in zip(symbol_names, state_vars))
    return (iterate,) + state_vars

  def aug_set_state(aug_loop_vars):
    # [FUNCTION DOC]
    # HELPER: splits the augmented state into the iterate (stored) and the user state (passed to
    # the user set-state callback). Returns nothing.
    nonlocal iterate
    # TODO(b/171479293): Drop the lint override.
    iterate, *loop_vars = aug_loop_vars  # pylint:disable=unused-variable
    # The iteration index is not "output" by the for loop. If the iterate
    # is used outside the loop, it will appear in the loop vars separately.
    set_state(loop_vars)

  def aug_body():
    # [FUNCTION DOC]
    # HELPER: loop body; runs the user body with the current iterate, then advances it by the
    # step. Returns nothing.
    nonlocal iterate
    body(iterate)
    iterate += delta

  def aug_test():
    # [FUNCTION DOC]
    # HELPER: loop condition; true while the iterate has not passed the limit in the direction of
    # the step, and, if given, the extra test holds. Returns a boolean tensor.
    # TODO(b/159713842): Remove once constant folding works.
    const_delta = tensor_util.constant_value(delta)
    if const_delta is not None:
      if const_delta >= 0:
        main_test = iterate < limit
      else:
        main_test = iterate > limit
    else:
      main_test = math_ops.logical_or(
          math_ops.logical_and(delta >= 0, iterate < limit),
          math_ops.logical_and(delta < 0, iterate > limit))

    if extra_test is not None:
      main_test = tf_cond.cond(main_test, extra_test, lambda: False)
    return main_test

  _add_max_iterations_hint(
      opts,
      math_ops.cast(misc.get_range_len(start, limit, delta), dtypes.int32))

  _tf_while_stmt(
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      ('<internal iterate>',) + symbol_names,
      opts)


def _tf_iterator_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  # [FUNCTION DOC]
  # HELPER: staged for loop over a distributed-style iterator.
  # Same inputs as the dispatcher. Adds a has-next flag to the loop state. Each iteration fetches
  # an optional next element; only if it has a value, a conditional runs the user body and returns
  # the new state (verified against the previous one), otherwise the state is kept. The loop runs
  # while the flag (and the extra test, if given) holds. Validates the initial state first and
  # returns nothing.
  """Overload of for_stmt that iterates over TF Iterators. See for_loop."""
  symbol_names = ('<internal has_next>',) + symbol_names
  has_next = True

  def aug_get_state():
    # [FUNCTION DOC]
    # HELPER: returns the has-next flag followed by the user state.
    return (has_next,) + get_state()

  def aug_set_state(aug_loop_vars):
    # [FUNCTION DOC]
    # HELPER: stores the first element as the has-next flag and passes the rest to the user
    # set-state callback. Returns nothing.
    nonlocal has_next
    # TODO(b/171479293): Drop the lint override.
    has_next, *loop_vars = aug_loop_vars  # pylint:disable=unused-variable
    set_state(loop_vars)

  init_vars = aug_get_state()
  verify_loop_init_vars(init_vars, symbol_names)

  def aug_body():
    # [FUNCTION DOC]
    # HELPER: loop body for the iterator loop; fetches the next optional element, records whether
    # it has a value, and conditionally runs the user body, storing the resulting state through
    # the augmented set-state. Returns nothing.
    """Main body passed to _tf_while_stmt."""
    nonlocal has_next
    opt_iterate = iter_.get_next_as_optional()
    has_next = opt_iterate.has_value()
    loop_vars = aug_get_state()  # updated by set_state() in _tf_while_loop.

    def main_path():
      # [FUNCTION DOC]
      # HELPER: conditional branch taken when an element exists; runs the user body on the value,
      # verifies the new state against the initial and entry state, and returns the new state.
      body(opt_iterate.get_value())
      new_loop_vars = aug_get_state()
      # Note: this verification duplicates the one performed in tf_while_stmt,
      # but needs to be done earlier to prevent the tf.cond from blowing up
      # first.
      verify_tf_loop_vars(
          init_vars, loop_vars, new_loop_vars, symbol_names, opts)
      return new_loop_vars

    def noop_path():
      # [FUNCTION DOC]
      # HELPER: conditional branch taken when the iterator is exhausted; returns the unchanged
      # entry state.
      return loop_vars

    # TODO(mdan): If tf.while_loop supported Optional, this could be avoided.
    # Calling set_state so that get_state() _tf_while_loop sees the conditional
    # tensors.
    aug_set_state(
        tf_cond.cond(has_next, main_path, noop_path))

  def aug_test():
    # [FUNCTION DOC]
    # HELPER: loop condition; the has-next flag (carried through the loop state), combined with
    # the extra test through a conditional when one is given. Returns a boolean tensor.
    # This value takes a complicated path to get here:
    #   prev_iteration_body -> get_state -> tf.while_loop (as loop var)
    #   -> current_iteration_body -> set_state -> has_next
    main_test = has_next
    if extra_test is not None:
      return tf_cond.cond(main_test, extra_test, lambda: False)
    return main_test

  _tf_while_stmt(
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      symbol_names,
      opts)


def _tf_distributed_iterable_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts):
  # [FUNCTION DOC]
  # HELPER: for loop over a distributed dataset/iterable, done with its reduce operation.
  # Same inputs as the dispatcher. Raises a not-implemented error if an extra test (break or
  # return) is given. Validates the initial state, converts shape invariants given as a mapping to
  # a positional list, then reduces over the iterable with the state as the accumulator: each step
  # restores the state, runs the user body on the element, reads and verifies the new state, and
  # returns it. The final result is passed to the set-state callback. Mutates opts; returns
  # nothing.
  """Overload of for_stmt that iterates over TF distributed datasets."""

  if extra_test is not None:
    raise NotImplementedError(
        'break and return statements are not yet supported in '
        'for ... in distributed input loops.')

  init_vars = get_state()
  verify_loop_init_vars(init_vars, symbol_names)

  if 'shape_invariants' in opts:
    opts['shape_invariants'] = _shape_invariants_mapping_to_positional_list(
        opts['shape_invariants'], init_vars)

  def reduce_body(loop_vars, iterate):
    # [FUNCTION DOC]
    # HELPER: reduction step. Inputs: loop_vars (current state) and iterate (next element).
    # Restores the state, runs the user body, verifies the new state against the initial and
    # current state, and returns the new state.
    set_state(loop_vars)
    body(iterate)
    new_loop_vars = get_state()
    verify_tf_loop_vars(
        init_vars, loop_vars, new_loop_vars, symbol_names, opts)
    return new_loop_vars

  set_state(iter_.reduce(init_vars, reduce_body))


def while_stmt(test, body, get_state, set_state, symbol_names, opts):
  # [FUNCTION DOC]
  # Functional form of a while statement; chooses staged or plain Python execution.
  # Inputs: test (callback returning the loop condition), body (callback for one iteration),
  # get_state / set_state (callbacks to read and restore the loop state), symbol_names (names of
  # the loop variables), opts (mapping of loop options).
  # The condition is first evaluated once inside a scratch function graph to decide the mode. If it
  # is a dense tensor, the staged loop runs (it re-evaluates the test). Otherwise, if the value is
  # false nothing runs; if true, one iteration is executed and the plain Python loop continues.
  # KNOWN ISSUE: the docstring claims a tuple with the final state is returned, but the function
  # returns nothing; results flow through the state callbacks.
  """Functional form of a while statement.

  The loop operates on a so-called state, which includes all symbols that are
  variant across loop iterations. In what follows we refer to state as either
  a tuple of entities that represent an actual state, or a list of arguments
  of the corresponding types.

  The inputs and outputs of the callables representing the loop blocks are not
  explicit - instead, these functions must use nonlocal/global for side effects.
  The inputs and outputs are instead controlled by the set_state/get_state
  functions.

  Args:
    test: Callable with boolean return type. The loop condition.
    body: Callable representing the actual loop body.
    get_state: Additional callable which can capture additional state (such as
      the values of composite symbols). This is only useful when staging the
      loop.
    set_state: Additional callable which save values captured by get_state back
      into the Python environment. This is only useful when staging the loop.
    symbol_names: Tuple containing the names of all loop variables.
    opts: Optional dict of extra loop parameters.

  Returns:
    Tuple containing the final state.
  """

  # Evaluate the initial test once in order to do the dispatch. The evaluation
  # is isolated to minimize unwanted side effects.
  # TODO(mdan): Do a full iteration - some state types might lower to Tensor.
  with func_graph.FuncGraph('tmp').as_default():
    init_test = test()

  # TensorFlow: Multiple evaluations are acceptable in this case, so we're fine
  # with the re-evaluation of `test` that `_tf_while_stmt` will make.
  if tensors.is_dense_tensor(init_test):
    _tf_while_stmt(test, body, get_state, set_state, symbol_names, opts)
    return

  # Normal Python: We already consumed one evaluation of `test`; consistently,
  # unroll one iteration before dispatching to a normal loop.
  # TODO(mdan): Push the "init_test" value via opts into _py_while_stmt?
  if not init_test:
    return
  body()

  _py_while_stmt(test, body, get_state, set_state, opts)


class _PythonLoopChecker(object):
  # [CLASS DOC]
  # HELPER: guard used by the plain-Python loop operators. Counts iterations (starting at one),
  # raises when a maximum iteration limit is exceeded, and, after many iterations, compares the
  # graph's operations before and after an iteration to warn once about a loop that appears to be
  # unrolling graph operations inefficiently. State: iteration count, flags for the unroll check
  # and for checking the op count after the iteration, and the set of operations seen before the
  # iteration (declared but only assigned when the check starts). Uses slots; relies on the module
  # constants for the limits and thresholds.
  """Verifies Python loops for TF-specific limits."""

  __slots__ = (
      'iterations',
      'check_inefficient_unroll',
      'check_op_count_after_iteration',
      'ops_before_iteration',
      )

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no inputs. Sets the iteration count to one, copies the module setting
    # that enables the inefficient-unroll warning, and clears the op-count check flag.
    # KNOWN ISSUE: the slot for the operations seen before an iteration is not initialized here,
    # so reading it before it is first assigned raises an attribute error.
    self.iterations = 1
    self.check_inefficient_unroll = WARN_INEFFICIENT_UNROLL

    # Triggered when we decided to test the op counts.
    self.check_op_count_after_iteration = False

  def _get_ops(self):
    # [FUNCTION DOC]
    # HELPER: returns the set of operations currently in the default graph.
    return set(ops.get_default_graph().get_operations())

  def _check_unroll_limits(self):
    # [FUNCTION DOC]
    # HELPER: raises a value error if the iteration count exceeds the module's maximum Python
    # iterations; otherwise returns nothing.
    if self.iterations > PYTHON_MAX_ITERATIONS:
      raise ValueError('iteration limit exceeded')

  def _stop_checking_inefficient_unroll(self):
    # [FUNCTION DOC]
    # HELPER: turns off all further unroll checking (both flags) and discards the saved operation
    # set. Returns nothing.
    self.check_inefficient_unroll = False
    self.check_op_count_after_iteration = False
    self.ops_before_iteration = None

  def _verify_inefficient_unroll(self):
    # [FUNCTION DOC]
    # HELPER: compares the graph's current operations with those saved before the iteration
    # (asserting the saved set exists). If at least the minimum number of new operations appeared,
    # logs a warning with the iteration count, the new operations and the current stack trace, and
    # returns true; otherwise returns false.
    """Checks for possibly-inefficient creation of ops in a Python loop."""
    assert self.ops_before_iteration is not None
    ops_after_iteration = self._get_ops()
    new_ops = tuple(
        op for op in ops_after_iteration if op not in self.ops_before_iteration)

    if len(new_ops) < INEFFICIENT_UNROLL_MIN_OPS:
      return False

    ag_logging.warning(
        'Large unrolled loop detected. Did you mean to use a TF loop?'
        ' The following ops were created after iteration %s: %s'
        '\nSee'
        ' https://github.com/tensorflow/tensorflow/blob/master/'
        'tensorflow/python/autograph/g3doc/reference/common_errors.md'
        '#warning-large-unrolled-loop-detected'
        '\n'
        'Location:'
        '\n%s'
        '', self.iterations, new_ops, '\n'.join(traceback.format_stack()))
    return True

  def before_iteration(self):
    # [FUNCTION DOC]
    # Hook called before each loop iteration. Once the iteration count passes the minimum
    # threshold and unroll checking is enabled, snapshots the graph's operations and arms the
    # post-iteration op-count check. Returns nothing.
    """Called before each iteration in a Python loop."""
    if (self.check_inefficient_unroll and
        self.iterations > INEFFICIENT_UNROLL_MIN_ITERATIONS):
      self.ops_before_iteration = self._get_ops()
      self.check_op_count_after_iteration = True

  def after_iteration(self):
    # [FUNCTION DOC]
    # Hook called after each loop iteration. Increments the iteration count and enforces the
    # maximum (raising a value error when exceeded). If the op-count check is armed, verifies
    # for inefficient unrolling: after a warning checking stops for good, and checking also stops
    # a few iterations after the threshold. Returns nothing.
    """Called after each iteration in a Python loop."""
    self.iterations += 1

    self._check_unroll_limits()

    if self.check_op_count_after_iteration:
      did_warn = self._verify_inefficient_unroll()
      if did_warn:
        self._stop_checking_inefficient_unroll()  # Only warn once.
      elif self.iterations > INEFFICIENT_UNROLL_MIN_ITERATIONS + 3:
        # Once deciding to check the op counts, only do it for a few iterations.
        self._stop_checking_inefficient_unroll()


def _py_while_stmt(test, body, get_state, set_state, opts):
  # [FUNCTION DOC]
  # HELPER: plain-Python implementation of the while statement operator.
  # Inputs as for the dispatcher; the state callbacks and options are ignored. With debug
  # assertions on, wraps the body so the loop checker is notified around each iteration. Repeatedly
  # evaluates the test, converting it to a boolean (a graph tensor error becomes a not-implemented
  # error explaining that the condition changed to a tensor mid-loop), and runs the body while
  # true. Returns nothing.
  """Overload of while_stmt that executes a Python while loop."""
  del opts, get_state, set_state

  if __debug__:
    checker = _PythonLoopChecker()
    before_iteration = checker.before_iteration
    after_iteration = checker.after_iteration
    before_iteration()

    original_body = body
    def protected_body():
      # [FUNCTION DOC]
      # HELPER: wraps the loop body for the loop checker: runs the original body, then reports
      # the end of the iteration and the start of the next. Returns nothing.
      original_body()
      after_iteration()
      before_iteration()
    body = protected_body

  def guarded_test():
    # [FUNCTION DOC]
    # HELPER: evaluates the loop condition and converts it to a boolean. A graph-tensor conversion
    # error is logged at verbosity 1 and re-raised as a not-implemented error (chained to the
    # cause) with guidance. Returns the boolean.
    test_result = test()
    try:
      # Note: Using try/except and not tensor_util.is_tf_type to avoid
      # performance degradation.
      return bool(test_result)
    except errors_impl.OperatorNotAllowedInGraphError as e:
      ag_logging.log(
          1,
          'Caught error while evaluating while loop condition',
          exc_info=True)
      # TODO(mdan): distinguish between these two cases.
      raise NotImplementedError(
          'The condition of while loop started as non-Tensor, then changed to'
          ' Tensor. This may happen either because variables changed type, or'
          ' when a break or return statement inside the loop depends on a'
          ' Tensor condition. In both cases, changing to a TF loop should'
          ' remove the error.\nSee '
          'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/'
          'python/autograph/g3doc/reference/limitations.md'
          '#consistency-of-control-flow-types for more info.') from e
  while guarded_test():
    body()


def _shape_invariants_mapping_to_positional_list(mapping, keys):
  # [FUNCTION DOC]
  # HELPER: converts shape invariants given as pairs of (variable, shape) into a tuple aligned
  # with keys. Inputs: mapping (iterable of such pairs; matching is by object identity since
  # variables may be unhashable) and keys (the loop variables in order). For each key, returns
  # its invariant if present, else a structure of null values mirroring the key's nesting.
  # The keys are not expected to be hashable.
  mapping = {id(k): (k, v) for k, v in mapping}
  result = []
  for k in keys:
    map_key, map_val = mapping.get(id(k), (None, None))
    result.append(
        map_val if map_key is k else nest.map_structure(lambda _: None, k))
  return tuple(result)


# Textual description of what a legal TF loop variable is. This description
# summarizes types that _placeholder_value below can handle. Keep the two
# together and in sync.
LEGAL_LOOP_TYPES = 'Tensor, int, float, bool or a list, tuple or dict thereof'


def _placeholder_value(like, shape_invariant, original=None):
  # [FUNCTION DOC]
  # HELPER: builds a dummy initial value for a loop variable that was undefined before the loop.
  # Inputs: like (the value produced by the first iteration), shape_invariant (user shape or
  # null), original (value before the loop, default null). Returns a pair of the placeholder and
  # a shape invariant (or null). Null and undefined markers give the original value; numbers and
  # booleans give zero of the same type; tensors give a zero tensor whose unknown dimensions are
  # zero or taken from the invariant, with the invariant returned when dimensions were dynamic;
  # lists, tuples and dicts are handled element-wise and repacked. Raises a type error for any
  # other type.
  # KNOWN ISSUE: the docstring says only the placeholder is returned, but a pair is returned.
  """Constructs a (dummy) placeholder value for a loop-initialized variable.

  Args:
    like: Any object. The value created by the first iteration of the loop. If a
      Python scalar, the placeholder will be the zero value of that type. If a
      Tensor, the placeholder will be a zero tensor of matching shape and dtype.
      If a list, dict or tuple, the placeholder will be an identical structure
      of placeholders.
    shape_invariant: The shape invariant specified by the user (or None, if
      nothing was specified) for the respective variable.
    original: Any object. The value of the variable prior to entering the loop.
      Typically, this is one of the special "Undefined" value, because that's
      when a placeholder is needed.

  Returns:
    Either a zero value of structure, shape and dtype matching 'like', or
    'original', if no such zero value could be created.
  """
  if like is None:
    return original, None

  elif isinstance(like, (variables.Undefined, variables.UndefinedReturnValue)):
    return original, None

  elif isinstance(like, (int, float, bool)):
    return type(like)(0), None

  elif tensor_util.is_tf_type(like):

    like_shape = shape_invariant if shape_invariant is not None else like.shape
    if like_shape is None or like_shape.rank is None:
      return array_ops.zeros((), like.dtype), like_shape

    # If the shape contains dynamic values, set the corresponding starting
    # dimension to either zero or what the shape invariant specified.
    placeholder_shape = []
    has_dynamic_dims = False
    for s, i in zip(like.shape, like_shape):
      if i is None:
        like_dim = 0
      elif isinstance(i, tensor_shape.Dimension):
        if i.value is None:
          like_dim = 0
        else:
          like_dim = i.value
      else:
        like_dim = i

      if s is None:
        placeholder_shape.append(like_dim)
        has_dynamic_dims = True
      elif isinstance(s, tensor_shape.Dimension):
        if s.value is None:
          placeholder_shape.append(like_dim)
          has_dynamic_dims = True
        else:
          placeholder_shape.append(s.value)
      else:
        placeholder_shape.append(s)

    if has_dynamic_dims:
      invariant = like_shape
    else:
      invariant = None

    return array_ops.zeros(placeholder_shape, like.dtype), invariant

  elif isinstance(like, (list, tuple, dict)):
    if shape_invariant is None:
      zipped = nest.map_structure(lambda v: _placeholder_value(v, None),
                                  nest.flatten(like))
    else:
      zipped = nest.map_structure(_placeholder_value, nest.flatten(like),
                                  nest.flatten(shape_invariant))
    vals, invars = zip(*zipped)
    return (nest.pack_sequence_as(like,
                                  vals), nest.pack_sequence_as(like, invars))

  # This is to be caught by _try_handling_undefineds, to give more context.
  raise TypeError(
      "Found an unsupported type '{}' while creating placeholder for {}."
      ' Supported types include Tensor, int, float, bool, list, tuple or dict.'
      .format(type(like).__name__, like))


def _try_handling_undefineds(body, get_state, set_state, init_vars, nulls,
                             shape_invariants, symbol_names):
  # [FUNCTION DOC]
  # HELPER: tries to replace null or undefined loop variables with placeholder values by staging
  # one iteration of the body in a scratch graph.
  # Inputs: body, get_state, set_state (loop callbacks), init_vars (values before the loop),
  # nulls (per-variable flags: null or undefined), shape_invariants (user invariant per variable),
  # symbol_names. The state is temporarily set to autocast values, the body is run once, and
  # placeholders are built for flagged variables from the first-iteration values. Errors of the
  # kinds unbound-local, type, value and key are logged and turned into a failure note. The state
  # is always restored to the (possibly updated) initial variables if it was modified. Finally,
  # validates the initial variables (this raises if some remain null or undefined).
  # Returns a tuple of success flag, new initial variables, and extra shape invariants.
  # KNOWN ISSUE: the docstring lists four returned items including a failure message, but only
  # three are returned; and on the failure path the success flag and the extra invariants are
  # never assigned, so returning would raise an unbound-variable error unless validation above
  # raised first.
  """Makes a best-effort attempt to substitute undefineds with placeholders.

  Note: this substitution requires two things to happen:
   1. the types of loop variables could be inferred (usually by staging one
       iteration)
   2. these types could be replaced by placeholders (e.g. zero values, for
       tensors).

  Args:
    body: a function representing the loop body. See while_stmt.
    get_state: state getter for the loop statement. See while_stmt.
    set_state: state getter for the loop statement. See while_stmt.
    init_vars: loop variables before entering the loop. See while_stmt.
    nulls: list of boolean flags indicating whether the corresponding loop var
      is None or undefined.
    shape_invariants: user-specified shape invariant for each loop variable.
    symbol_names: list of loop variable names. See while_stmt.

  Returns:
    A tuple (success, new_init_vars, extra_shape_invariants, failure_message):
     * success is a boolean flag indicating
       whether types could be successfully inferred (step 1 above)
     * new_init_vars contains the loop vars, with None or undefined values
       replaced by default values, where possible (step 2 above)
     * extra_shape_invariants contains shape invariants that would be needed
       by while_stmt, for instance if the placeholder values had a shape
       different from the corresponding loop outputs
  """
  state_modified = False
  first_iter_vars = None
  failure_message = None

  try:
    # Stage an iteration of the loop body in a temporary graph.
    with func_graph.FuncGraph('tmp').as_default():
      # This call to set_state helps report nicer error messages when symbols
      # are inconsistently used.
      # Another complication is that non_tensor values will be autocast to
      # Tensor by while_loop, and their static value lost. So we need to account
      # that here.
      def autocast_to_tensor(v):
        # [FUNCTION DOC]
        # HELPER: converts Python scalars, strings, lists, tuples and numpy values into a
        # placeholder with the dtype and shape they would have as tensors (mimicking the loop
        # op's conversion, which loses static values); returns other values unchanged.
        if isinstance(
            v, (int, float, bool, str, list, tuple, np.ndarray, np.generic)):
          init_val = tensor_conversion.convert_to_tensor_v2(v)
          return array_ops.placeholder(init_val.dtype, init_val.shape)
        return v
      autocast_init_vars = nest.map_structure(autocast_to_tensor, init_vars)
      set_state(autocast_init_vars)
      state_modified = True

      body()
      first_iter_vars = get_state()

    # Note: the actual placeholder value doesn't matter, because as the
    # staging proved, it will be replaced by an actual value before being
    # read.
    inits_and_invariants = tuple(
        (_placeholder_value(iv, i, v) if n else (v, None))
        for v, n, iv, i in zip(init_vars, nulls, first_iter_vars,
                               shape_invariants))
    init_vars, extra_shape_invariants = zip(*inits_and_invariants)
    success = True

  except (UnboundLocalError, TypeError, ValueError, KeyError):
    ag_logging.log(1, 'Caught error while staging loop body', exc_info=True)
    # Fall back to the old functionality. It will likely result in an input
    # validation failure.
    exc = sys.exc_info()
    failure_message = (
        'Note: AutoGraph tried to define it automatically, but ran into a'
        ' {}: {}'.format(exc[0].__name__, exc[1]))

  finally:
    if state_modified:
      set_state(init_vars)

  # This check runs regardless, in case we captured non-Tensor inputs.
  verify_loop_init_vars(
      init_vars, symbol_names, first_iter_vars, extra_message=failure_message)

  return success, init_vars, extra_shape_invariants


def _runtime_zero_iterations_errmsg(symbol_names, nulls, init_vars):
  # [FUNCTION DOC]
  # HELPER: builds the runtime error text used when a loop with placeholder-initialized variables
  # runs zero times. Inputs: symbol_names, nulls (flags marking placeholder variables), init_vars
  # (original values). Lists the flagged names, using a generic phrase for the function return
  # value. Returns the message text.
  """Creates an error message asking for the loop to iterate at least once."""
  var_names = []
  for sn, n, v in zip(symbol_names, nulls, init_vars):
    if not n:
      continue
    if isinstance(v, variables.UndefinedReturnValue):
      var_names.append('the function return value')
    else:
      var_names.append(sn)
  var_names = ', '.join(var_names)
  return 'loop must iterate at least once to initialize {}'.format(var_names)


def _tf_while_stmt(test, body, get_state, set_state, symbol_names, opts):
  # [FUNCTION DOC]
  # HELPER: staged implementation of the while statement operator (also used by the for loops).
  # Inputs as for the dispatcher. Reads the initial state; if any variable is null or undefined,
  # tries to substitute placeholders (merging remapped shape invariants into the options). When
  # placeholders are used, the loop carries an extra leading flag recording that at least one
  # iteration ran, and afterwards asserts it at graph run time, with an error message asking for
  # at least one iteration. The wrapped test and body restore the state from the loop variables,
  # call the user callbacks, verify the new variables and return them. Shape invariants given as a
  # mapping are converted to a positional list (this mutates opts), iterate names are dropped from
  # the loop options, and the loop is built with a flag to keep the structure of results. The final
  # variables (flag removed, tensors passed through identity under the assertion's dependency) are
  # passed to the set-state callback. Returns nothing.
  """Overload of while_stmt that stages a TF while_stmt."""
  init_vars = get_state()
  orig_init_vars = init_vars

  nulls = tuple(_is_none_or_undef(v) for v in init_vars)
  if any(nulls):
    shape_invars_by_init_vals = {
        id(v): i for v, i in opts.get('shape_invariants', ())
    }
    shape_invariants = tuple(
        shape_invars_by_init_vals.get(id(v), None) for v in orig_init_vars)
    (require_one_iteration, init_vars,
     extra_shape_invariants) = _try_handling_undefineds(body, get_state,
                                                        set_state, init_vars,
                                                        nulls, shape_invariants,
                                                        symbol_names)
  else:
    require_one_iteration = False

  if require_one_iteration:
    merged_shape_invariants = dict(shape_invars_by_init_vals)
    # This has two roles:
    #  1. Shape invariants are remapped from the old init vars to the new ones.
    #  2. Any new shape invariants created by the init vars are kept, but only
    #     if the user didn't already specify some.
    for v, nv, ni in zip(orig_init_vars, init_vars, extra_shape_invariants):
      merged_invariant = merged_shape_invariants.get(id(v), ni)
      if merged_invariant is not None:
        merged_shape_invariants[id(nv)] = merged_invariant
    merged_shape_invariants = tuple((nv, merged_shape_invariants[id(nv)])
                                    for nv in init_vars
                                    if id(nv) in merged_shape_invariants)
    if merged_shape_invariants:
      opts = dict(**opts)
      opts['shape_invariants'] = merged_shape_invariants

  def aug_test(*loop_vars):
    # [FUNCTION DOC]
    # HELPER: loop condition for the staged loop. Takes the loop variables (dropping the leading
    # run flag if present), restores them into the state, evaluates the user test and returns it
    # verified as a scalar boolean tensor.
    if require_one_iteration:
      loop_vars = loop_vars[1:]

    set_state(loop_vars)
    return _verify_tf_condition(test(), 'while loop')

  def aug_body(*loop_vars):
    # [FUNCTION DOC]
    # HELPER: loop body for the staged loop. Takes the loop variables (dropping the leading run
    # flag if present), restores them into the state, runs the user body, reads and verifies the
    # new variables against the initial and entry ones, and returns them (with the run flag set to
    # true prepended if used).
    if require_one_iteration:
      loop_vars = loop_vars[1:]

    set_state(loop_vars)
    body()
    new_loop_vars = get_state()
    verify_tf_loop_vars(
        init_vars, loop_vars, new_loop_vars, symbol_names, opts)

    if require_one_iteration:
      new_loop_vars = (True,) + new_loop_vars

    return new_loop_vars

  if 'shape_invariants' in opts:
    opts['shape_invariants'] = _shape_invariants_mapping_to_positional_list(
        opts['shape_invariants'], init_vars)

  while_loop_opts = dict(opts)
  while_loop_opts.pop('iterate_names', None)

  # Non-v2 while_loop unpacks the results when there is only one return value.
  # This enforces consistency across versions.
  while_loop_opts['return_same_structure'] = True

  if require_one_iteration:
    aug_init_vars = (False,) + init_vars
    if 'shape_invariants' in while_loop_opts:
      while_loop_opts['shape_invariants'] = (
          (None,) + while_loop_opts['shape_invariants'])
  else:
    aug_init_vars = init_vars

  final_loop_vars = while_loop.while_loop(aug_test, aug_body, aug_init_vars,
                                          **while_loop_opts)

  if require_one_iteration:
    with ops.control_dependencies([
        control_flow_assert.Assert(final_loop_vars[0], [
            _runtime_zero_iterations_errmsg(symbol_names, nulls, orig_init_vars)
        ])
    ]):
      final_loop_vars = nest.map_structure(
          lambda v: (array_ops.identity(v) if tensor_util.is_tf_type(v) else v),
          final_loop_vars[1:],
      )

  set_state(final_loop_vars)


def if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts):
  # [FUNCTION DOC]
  # Functional form of an if statement; chooses staged or plain Python execution.
  # Inputs: cond (condition; a dense tensor or a Python value), body and orelse (callbacks for the
  # two blocks), get_state / set_state (callbacks to read and restore the state), symbol_names
  # (names of the state variables), nouts (how many leading state variables are outputs).
  # A dense tensor condition (sparse tensors are not supported by the conditional) uses the staged
  # version; anything else uses the plain version, which ignores the state arguments. Returns
  # nothing; results flow through the state callbacks.
  """Functional form of an if statement.

  The conditional operates on a state, which includes all symbols whose values
  are a function of the branch taken.

  For example, given the code below that calculates the abs function:

  ```
    x = 1
    if x > 0:
      x = -x
  ```

  The state is represented by the variable `x`. The `body, `orelse` and
  `set_state` functions must bind to the original `x` symbol, using `nonlocal`.

  The inputs and outputs of the callables representing the loop blocks are not
  explicit - instead, these functions must use nonlocal/global for side effects.
  The inputs and outputs are instead controlled by the set_state/get_state
  functions.

  Args:
    cond: Boolean.
    body: Callable representing the main block of the conditional.
    orelse: Callable representing the else block of the conditional.
    get_state: Function that returns a tuple containing the values of all
      composite symbols modified within the conditional. This allows access to
      state that branches may mutate through side effects. This function is not
      needed and should not be called when dispatching to code matching Python's
      default semantics. This is useful for checkpointing to avoid unintended
      side-effects when staging requires evaluating all code-paths.
    set_state: Function to set the values of all composite symbols modified
      within the conditional. This is the complement to get_state, used to
      restore checkpointed values. The single argument a tuple containing values
      for each composite symbol that may be modified in a branch of the
      conditional. The is usually the result of a call to get_state.
    symbol_names: Tuple containing basic loop var names.
    nouts: Number of variables output by the statement. Vars which are not
      outputs will not be passed through staged control flow such as tf.cond.
      This includes variables that are defined before the conditional, but are
      not used after it.
  """
  # Note: tf.cond doesn't support SparseTensor.
  if tensors.is_dense_tensor(cond):
    _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
  else:
    _py_if_stmt(cond, body, orelse)


def _tf_if_stmt(
    cond, body, orelse, get_state, set_state, symbol_names, nouts):
  # [FUNCTION DOC]
  # HELPER: staged implementation of the if statement operator.
  # Same inputs as the dispatcher. Verifies the condition is a scalar boolean tensor. If there are
  # no outputs, wraps the state callbacks to add a dummy leading value (the older control flow
  # needs at least one output) and adds a dummy name. Each branch restores the initial state, runs
  # its block, keeps the first nouts values, checks they are defined, and checks them against the
  # other branch's values when available. The conditional is built in strict mode; non-output
  # state is appended unchanged and the final state is passed to set-state. Returns nothing.
  """Overload of if_stmt that stages a TF cond."""
  cond = _verify_tf_condition(cond, 'if statement')

  if not nouts:
    prev_get_state, prev_set_state = get_state, set_state
    # Control flow V1 wants at least one output.
    get_state = lambda: (0,) + prev_get_state()
    set_state = lambda v: prev_set_state(v[1:])
    symbol_names += ('<unused dummy>',)
    nouts = 1

  init_vars = get_state()

  # TODO(mdan): Use nonlocal once we no longer need to support py2.
  new_body_vars_ = [None]
  new_orelse_vars_ = [None]

  def aug_body():
    # [FUNCTION DOC]
    # HELPER: main branch of the staged conditional; restores the initial state, runs the body,
    # records and validates the output values, cross-checks with the else branch if already
    # recorded, and returns the output values.
    set_state(init_vars)
    body()
    new_body_vars = get_state()
    new_body_vars = new_body_vars[:nouts]
    new_body_vars_[0] = new_body_vars
    _verify_tf_cond_branch_vars(new_body_vars, symbol_names, 'main')
    if new_orelse_vars_[0] is not None:
      _verify_tf_cond_vars(new_body_vars, new_orelse_vars_[0], symbol_names)
    return new_body_vars

  def aug_orelse():
    # [FUNCTION DOC]
    # HELPER: else branch of the staged conditional; restores the initial state, runs the else
    # block, records and validates the output values, cross-checks with the main branch if already
    # recorded, and returns the output values.
    set_state(init_vars)
    orelse()
    new_orelse_vars = get_state()
    new_orelse_vars = new_orelse_vars[:nouts]
    new_orelse_vars_[0] = new_orelse_vars
    _verify_tf_cond_branch_vars(new_orelse_vars, symbol_names, 'else')
    if new_body_vars_[0] is not None:
      _verify_tf_cond_vars(new_body_vars_[0], new_orelse_vars, symbol_names)
    return new_orelse_vars

  final_cond_vars = tf_cond.cond(
      cond, aug_body, aug_orelse, strict=True)
  final_cond_vars = final_cond_vars + init_vars[nouts:]

  set_state(final_cond_vars)


def _py_if_stmt(cond, body, orelse):
  # [FUNCTION DOC]
  # HELPER: plain-Python implementation of the if statement operator; calls and returns the
  # result of the body callback if cond is truthy, otherwise of the else callback.
  """Overload of if_stmt that executes a Python if statement."""
  return body() if cond else orelse()
