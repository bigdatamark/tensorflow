# [MODULE DOC]
# AutoGraph overloads of Python builtin functions (abs, float, int, len, print, min, max, range,
# enumerate, zip, map, next, filter, any, all, sorted) plus helpers for evaluating locals, globals,
# eval and super in the frame of the original (unconverted) function.
# Each public overload (name ending in an underscore) dispatches either to a user-registered
# override found via a per-builtin type registry, to a TensorFlow implementation when an argument
# is a TF tensor, or to the plain Python builtin.
# Import-time side effects: creates a module-level sentinel for "unspecified" arguments, one empty
# type registry per overloadable builtin (extensible by other modules), the tuple of supported
# builtins and the mapping from builtin names to their overloads.
# Used by the converted code produced by the AutoGraph call-trees / builtin-functions converters.
#
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
"""Operators corresponding to Python builtin functions.

List of built-in functions: https://docs.python.org/3/library/functions.html
"""

import inspect

from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops


UNSPECIFIED = object()

abs_registry = type_registry.TypeRegistry()
len_registry = type_registry.TypeRegistry()
print_registry = type_registry.TypeRegistry()
enumerate_registry = type_registry.TypeRegistry()
zip_registry = type_registry.TypeRegistry()
map_registry = type_registry.TypeRegistry()
filter_registry = type_registry.TypeRegistry()
any_registry = type_registry.TypeRegistry()
all_registry = type_registry.TypeRegistry()
sorted_registry = type_registry.TypeRegistry()
next_registry = type_registry.TypeRegistry()


def registry_lookup(reg, obj):
  # [FUNCTION DOC]
  # HELPER: shared by all builtin overloads to find a user-registered override.
  # Looks up the type of obj in the type registry reg. Returns the registered override callable,
  # or the null value if no registered type matches (the lookup error is swallowed).
  try:
    return reg.lookup(obj)
  except LookupError:
    pass
  return None


def overload_of(f):
  # [FUNCTION DOC]
  # Maps a Python builtin function f to its AutoGraph overload. If f is one of the supported
  # builtins, returns the overload registered under the builtin's name; otherwise returns f
  # unchanged. Has no side effects.
  if f in SUPPORTED_BUILTINS:
    return BUILTIN_FUNCTIONS_MAP[f.__name__]
  return f


def _find_originating_frame(caller_fn_scope, innermost=True):
  # [FUNCTION DOC]
  # HELPER: used by the in-original-context evaluators below to find the right stack frame.
  # Walks the call stack outward from the current frame looking for a frame whose local variables
  # contain the function-scope object caller_fn_scope under its own name. If innermost is true
  # (default) the first match is returned; otherwise the outermost match is returned.
  # Raises an assertion error if no frame matches.
  """Locates the frame in which `caller_fn_scope` was defined."""
  ctx_frame = inspect.currentframe()
  result = None
  while ctx_frame is not None:
    # Note it should not be normally possible to get false positives this way
    # because the function scope object is not accessible to user code (barring
    # call stack introspection).
    if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
      result = ctx_frame
      if innermost:
        break
    ctx_frame = ctx_frame.f_back

  assert result is not None, (
      'the conversion process should ensure the caller_fn_scope is always'
      ' found somewhere on the call stack')

  return result


def locals_in_original_context(caller_fn_scope):
  # [FUNCTION DOC]
  # Returns the mapping of local variable names to values of the innermost stack frame that
  # belongs to the function scope caller_fn_scope. Does not execute anything despite the docstring.
  # Raises an assertion error if the scope is not on the call stack.
  """Executes the locals function in the context of a specified function."""
  return _find_originating_frame(caller_fn_scope, innermost=True).f_locals


def globals_in_original_context(caller_fn_scope):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring is copied from the locals variant and says it executes the locals
  # function; it actually returns the global-variable mapping of the frame.
  # Returns the mapping of global names to values of the innermost stack frame that belongs to the
  # function scope caller_fn_scope. Raises an assertion error if the scope is not on the call stack.
  """Executes the locals function in the context of a specified function."""
  return _find_originating_frame(caller_fn_scope, innermost=True).f_globals


def eval_in_original_context(f, args, caller_fn_scope):
  # [FUNCTION DOC]
  # Calls the eval-like callable f with the original call arguments args (a sequence whose first
  # item is the expression to evaluate), filling in missing globals (second item) and locals
  # (third item) from the innermost frame belonging to the function scope caller_fn_scope.
  # Returns whatever f returns. Raises an assertion error if the scope is not on the call stack.
  """Executes the eval function in the context of a specified function."""
  # When control flow is rewritten using functions, eval should use the
  # variables found in the same block where it was called. That is equivalent
  # to the innermost function call.
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)

  args = (
      args[0],
      ctx_frame.f_globals if len(args) < 2 else args[1],
      ctx_frame.f_locals if len(args) < 3 else args[2],
  )
  return f(*args)


def super_in_original_context(f, args, caller_fn_scope):
  # [FUNCTION DOC]
  # Calls the super callable f. If explicit arguments args were given they are passed through
  # unchanged. For the zero-argument form, it finds the outermost frame of the function scope
  # caller_fn_scope, reads the implicit class cell variable and the first positional argument
  # (normally self) from that frame, and calls f with those two values.
  # Returns the super object. Raises an assertion error if the scope is not on the call stack and
  # a key error if the frame has no class cell variable.
  """Executes the super function in the context of a specified function.

  See https://docs.python.org/3/library/functions.html#super for the exact
  details

  Args:
    f: Callable, typically the super builtin
    args: List[Any], the original call arguments
    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
      scope of the converted function in which this call was originally made

  Returns:
    The result of calling `f` as if it was called in the frame indicated by
      `caller_fn_scope`.
  """

  # Only the no-arg call is desugared.
  if args:
    return f(*args)

  # Inner functions seem to include their closure in f_locals, so we need
  # to find the outermost frame.
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)

  # When super(..) is called without arguments, it looks for __class__ cell
  # variable and the first argument passed in the enclosing function according
  # to the spec https://www.python.org/dev/peps/pep-3135/ .
  #
  # We couldn't verify if `inspect.currentframe().f_code.co_varnames[0]` is
  # guaranteed to be the first argument from an official doc or PEP, however,
  # it's fairly stable and well established:
  # - An unofficial community doc mentions it.
  #   https://python-reference.readthedocs.io/en/latest/docs/code/varnames.html
  # - CPython has tests checking that order, which was merged in 2008, and
  #   unchanged since then.
  #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py2_test_grammar.py#L157
  #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py3_test_grammar.py#L192
  #
  # Note: the name can be more reliably obtained by inspecting the calling
  # function's argspec.
  #
  # Even though methods can be declared using *args (def method(*args)),
  # that pattern is disallowed by super() -- it raises super() no arguments.
  # Method definitions using **kwargs are not allowed at all.
  # In other words, we can always assume that self is on the first positional
  # argument (for correct code).
  #
  # TODO(mdan): Consider additional checks in case the input code is incorrect.
  # For example, the error might be cryptic compared to what super() regularly
  # raises.

  type_arg = ctx_frame.f_locals['__class__']
  self_arg_name = ctx_frame.f_code.co_varnames[0]
  self_arg = ctx_frame.f_locals[self_arg_name]
  return f(type_arg, self_arg)


def abs_(x):
  # [FUNCTION DOC]
  # Overload of abs. Uses a registered override for the type of x if any; otherwise uses the
  # TensorFlow absolute value op for tensors and the Python abs for everything else.
  abs_override = registry_lookup(abs_registry, x)
  if abs_override is not None:
    return abs_override(x)
  if tensor_util.is_tf_type(x):
    return _tf_abs(x)
  return _py_abs(x)


def _tf_abs(x):
  # [FUNCTION DOC]
  # HELPER: tensor branch of abs_.
  # Returns the element-wise absolute value of tensor x using the TensorFlow math op.
  return math_ops.abs(x)


def _py_abs(x):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of abs_.
  # Returns the Python absolute value of x.
  return abs(x)


def float_(x=0):
  # [FUNCTION DOC]
  # Overload of float. Argument x defaults to zero. Tensors are converted with the TensorFlow
  # path (_tf_float); anything else uses the Python float conversion. Returns the converted value.
  if tensor_util.is_tf_type(x):
    return _tf_float(x)
  return _py_float(x)


def _tf_float(x):
  # [FUNCTION DOC]
  # HELPER: tensor branch of float_.
  # If tensor x has string dtype, parses its elements as numbers into 32-bit float; otherwise
  # casts x to 32-bit float. Always 32-bit regardless of the input precision (see the TODO).
  # TODO(mdan): We shouldn't assume float32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.float32)
  return math_ops.cast(x, dtype=dtypes.float32)


def _py_float(x):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of float_.
  # Returns x converted with the Python float conversion.
  return float(x)


def int_(x=0, base=UNSPECIFIED):
  # [FUNCTION DOC]
  # Overload of int. x defaults to zero; base is an optional integer radix (unspecified sentinel
  # by default). Tensors go through _tf_int, everything else through the Python conversion.
  # Returns the converted value.
  if tensor_util.is_tf_type(x):
    return _tf_int(x, base)
  return _py_int(x, base)


def _tf_int(x, base):
  # [FUNCTION DOC]
  # HELPER: tensor branch of int_.
  # Only base ten (or unspecified) is supported; any other base raises a not-implemented error.
  # String tensors are parsed into 32-bit integers; other tensors are cast to 32-bit integer
  # (always 32-bit, see the TODO). Returns the resulting tensor.
  if base not in (10, UNSPECIFIED):
    raise NotImplementedError('base {} not supported for int'.format(base))

  # TODO(mdan): We shouldn't assume int32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.int32)
  return math_ops.cast(x, dtype=dtypes.int32)


def _py_int(x, base):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of int_.
  # Converts x to an integer, passing base to the Python conversion only when it was specified.
  if base is UNSPECIFIED:
    return int(x)
  return int(x, base)


def len_(s):
  # [FUNCTION DOC]
  # Overload of len. Dispatch order: registered override for the type of s; tensor array (size
  # op); tensor list (list length op); other tensor (static or dynamic first-dimension size);
  # otherwise the Python len. Returns an integer or a scalar tensor.
  len_override = registry_lookup(len_registry, s)
  if len_override is not None:
    return len_override(s)
  if tensors.is_tensor_array(s):
    return _tf_tensor_array_len(s)
  elif tensors.is_tensor_list(s):
    return _tf_tensor_list_len(s)
  elif tensor_util.is_tf_type(s):
    return _tf_tensor_len(s)
  return _py_len(s)


def _tf_tensor_array_len(s):
  # [FUNCTION DOC]
  # HELPER: tensor-array branch of len_.
  # Returns the size of tensor array s as a scalar tensor.
  return s.size()


def _tf_tensor_list_len(s):
  # [FUNCTION DOC]
  # HELPER: tensor-list branch of len_.
  # Returns the number of elements of tensor list s as a scalar tensor.
  return list_ops.tensor_list_length(s)


def _tf_tensor_len(s):
  # [FUNCTION DOC]
  # HELPER: tensor branch of len_.
  # Returns the length of the first dimension of tensor s. If the first dimension is statically
  # known, returns it as a Python integer. If only the rank is known, uses the dynamic shape and
  # raises a value error if the tensor is a scalar. If the rank is unknown, builds a conditional
  # that returns the dynamic first dimension when the rank is positive and otherwise asserts
  # (failing at run time with a message) and yields zero.
  """Overload of len_ for Tensor arguments."""
  # Statically shaped tensors: length is known ahead of time.
  if s.shape.ndims and s.shape.dims[0].value is not None:
    return s.shape.dims[0].value

  # Static shape of unknown dimensions: use dynamic shape but statically
  # check that it's a scalar.
  shape = array_ops.shape(s)

  assert shape.shape, 'shape tensor of zero size? {}'.format(shape)

  if shape.shape[0] == 0:
    raise ValueError(
        'len requires a non-scalar tensor, got one of shape {}'.format(shape))

  if shape.shape.dims[0].value is not None:
    return array_ops.shape(s)[0]

  # Fully dynamic shape: use ops.
  rank = array_ops.rank(s)

  def raise_zero_rank_error():
    # [FUNCTION DOC]
    # HELPER: false branch of the rank check in the tensor length overload.
    # Builds an error message containing the dynamic rank, then, under a control dependency on a
    # failing assertion carrying that message, returns a constant zero. At run time this fails.
    msg = gen_string_ops.string_join(
        ['len requires non-zero rank, got ',
         gen_string_ops.as_string(rank)])
    with ops.control_dependencies([control_flow_assert.Assert(False, [msg])]):
      return constant_op.constant(0, dtype=dtypes.int32)

  return cond.cond(rank > 0, lambda: array_ops.shape(s)[0],
                   raise_zero_rank_error)


def _py_len(s):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of len_.
  # Returns the Python length of s.
  return len(s)


def print_(*objects, **kwargs):
  # [FUNCTION DOC]
  # Overload of print. Takes any number of positional objects and the keyword arguments separator,
  # end, file and flush; any other keyword raises a value error. If any object has a registered
  # print override, the first one found is used to print everything (this is how tensors get
  # printed in graphs); otherwise the plain Python print is used, after asserting that none of the
  # objects is a TensorFlow value (failure means the print op support module was not imported).
  # Returns whatever the chosen print function returns (nothing for the Python print, possibly an
  # op for an override).
  """Overload of the print builtin."""
  # Note: Python 2.6 doesn't support explicit keywords after starargs.
  unknown_kwargs = tuple(
      set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
  if unknown_kwargs:
    raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))

  print_fn = _py_print
  for x in objects:
    print_override = registry_lookup(print_registry, x)
    if print_override is not None:  # pylint: disable=comparison-with-callable
      print_fn = print_override
      break

  if print_fn is _py_print:
    # If this fails, ops/autograph_ops.py hasn't been imported.
    assert not any(tensor_util.is_tf_type(s) for s in objects)

  return print_fn(*objects, **kwargs)


def _py_print(*objects, **kwargs):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of print_.
  # Prints the objects with the given keyword arguments using the Python print. Side effect: writes
  # to the output stream. Returns nothing.
  print(*objects, **kwargs)


def min_(*args, **kwargs):
  # [FUNCTION DOC]
  # Overload of min. If any positional argument is a TensorFlow value uses the tensor path,
  # otherwise the Python min. Keyword arguments are forwarded. Returns the minimum.
  if any(tensor_util.is_tf_type(s) for s in args):
    return _tf_min(*args, **kwargs)
  return _py_min(*args, **kwargs)


def _tf_min(*args, **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: in the multi-argument error message the value reported as "shape" is the rank of
  # the offending tensor, not its shape.
  # HELPER: tensor branch of min_.
  # Any keyword argument raises a value error. With one argument: a scalar tensor is returned
  # as is, a rank-one tensor is reduced with a minimum over its axis, other ranks raise a value
  # error. With several arguments, every one must be a scalar tensor (else value error) and the
  # minimum over the stacked values is returned as a scalar tensor.
  if len(kwargs):
    kwargs_tuple = tuple(set(kwargs.keys()))
    raise ValueError('These keyword arguments are '
                     'currently not supported: {}'.format(kwargs_tuple))
  if len(args) == 1:
    rank = args[0].shape.rank
    if rank == 0:
      return args[0]
    if rank == 1:
      return math_ops.reduce_min(*args, axis=0)
    raise ValueError('min(arg) currently support only tensor with rank 1, '
                     'but got a tensor with rank {}'.format(rank))
  for arg in args:
    rank = arg.shape.rank
    if rank != 0:
      raise ValueError('min(arg1, arg2, *args) currently support '
                       'only scalar tensor, but got a tensor '
                       'with shape {}'.format(rank))
  return math_ops.reduce_min(args, axis=0)


def _py_min(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of min_.
  # Returns the result of the Python min called with the same arguments.
  return min(*args, **kwargs)


def max_(*args, **kwargs):
  # [FUNCTION DOC]
  # Overload of max. If any positional argument is a TensorFlow value uses the tensor path,
  # otherwise the Python max. Keyword arguments are forwarded. Returns the maximum.
  if any(tensor_util.is_tf_type(s) for s in args):
    return _tf_max(*args, **kwargs)
  return _py_max(*args, **kwargs)


def _tf_max(*args, **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: in the multi-argument error message the value reported as "shape" is the rank of
  # the offending tensor, not its shape.
  # HELPER: tensor branch of max_.
  # Any keyword argument raises a value error. With one argument: a scalar tensor is returned
  # as is, a rank-one tensor is reduced with a maximum over its axis, other ranks raise a value
  # error. With several arguments, every one must be a scalar tensor (else value error) and the
  # maximum over the stacked values is returned as a scalar tensor.
  if len(kwargs):
    kwargs_tuple = tuple(set(kwargs.keys()))
    raise ValueError('These keyword arguments are '
                     'currently not supported: {}'.format(kwargs_tuple))
  if len(args) == 1:
    rank = args[0].shape.rank
    if rank == 0:
      return args[0]
    if rank == 1:
      return math_ops.reduce_max(*args, axis=0)
    raise ValueError('max(arg) currently support only tensor with rank 1, '
                     'but got a tensor with rank {}'.format(rank))
  for arg in args:
    rank = arg.shape.rank
    if rank != 0:
      raise ValueError('max(arg1, arg2, *args) currently support '
                       'only scalar tensor, but got a tensor '
                       'with shape {}'.format(rank))
  return math_ops.reduce_max(args, axis=0)


def _py_max(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of max_.
  # Returns the result of the Python max called with the same arguments.
  return max(*args, **kwargs)


def range_(start_or_stop, stop=UNSPECIFIED, step=UNSPECIFIED):
  # [FUNCTION DOC]
  # Overload of range. start_or_stop is the stop value when it is the only argument, otherwise the
  # start; stop and step are optional (unspecified sentinel by default). If any of the three is a
  # TensorFlow value, returns a range tensor; otherwise a Python range object.
  if any(tensor_util.is_tf_type(s) for s in (start_or_stop, stop, step)):
    return _tf_range(start_or_stop, stop, step)
  return _py_range(start_or_stop, stop, step)


def _tf_range(start_or_stop, stop, step):
  # [FUNCTION DOC]
  # HELPER: tensor branch of range_.
  # Builds a range tensor. With a step, passes all values straight to the range op. With only a
  # stop, the stop is first clamped to be at least the start so that an empty range results
  # instead of a graph-construction error. With one argument, it is clamped to be at least zero.
  """Overload of range_ that generates a TF range tensor."""
  # Note: for static inputs (e.g. constants), tf.range errors out at graph
  # construction time, instead of returning an empty tensor. Preventing the
  # graph construction error aligns the semantics with Python.

  # TODO(mdan): We should optimize this when a full tensor is not required.
  if step is not UNSPECIFIED:
    # TODO(mdan): Add argument coercion similar to other cases.
    return math_ops.range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    stop = math_ops.maximum(start_or_stop, stop)
    return math_ops.range(start_or_stop, stop)
  start_or_stop = math_ops.maximum(start_or_stop, 0)
  return math_ops.range(start_or_stop)


def _py_range(start_or_stop, stop, step):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of range_.
  # Builds a Python range from one, two or three arguments depending on which of stop and step
  # were specified.
  if step is not UNSPECIFIED:
    return range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    return range(start_or_stop, stop)
  return range(start_or_stop)


def enumerate_(s, start=0):
  # [FUNCTION DOC]
  # Overload of enumerate. Uses a registered override for the type of s if any (called with s and
  # start); otherwise the Python enumerate. start defaults to zero. Returns the enumeration.
  enumerate_override = registry_lookup(enumerate_registry, s)
  if enumerate_override is not None:
    return enumerate_override(s, start)
  return _py_enumerate(s, start)


def _py_enumerate(s, start=0):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of enumerate_.
  # Returns the Python enumeration of s starting at start.
  return enumerate(s, start)


def zip_(*iterables, strict=False):
  # [FUNCTION DOC]
  # Overload of zip. Takes iterables and a keyword-only strict flag (default false). A registered
  # override is used only if every iterable has one and all overrides are the same function;
  # otherwise the plain Python zip is used. Returns the zipped result, passing strict through.
  zip_fn = _py_zip
  # If the overridden function is not the same across all iterables, use _py_zip
  for x in iterables:
    zip_override = registry_lookup(zip_registry, x)
    if zip_override is None or (zip_fn != _py_zip and zip_override != zip_fn):  # pylint: disable=comparison-with-callable
      zip_fn = _py_zip
      break
    zip_fn = zip_override
  return zip_fn(*iterables, strict=strict)


def _py_zip(*iterables, strict=False):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of zip_.
  # Returns a Python zip of the iterables; the strict flag is only forwarded when true (older
  # Python versions lack it). With strict, unequal lengths raise a value error when consumed.
  if strict:
    return zip(*iterables, strict=True)
  else:
    # Python < 3.10 doesn't have `strict` kwarg.
    return zip(*iterables)


def map_(fn, *iterables):
  # [FUNCTION DOC]
  # Overload of map. fn is the function to apply; iterables are the inputs. A registered override
  # is used only if every iterable has one and all overrides are the same function; otherwise the
  # plain Python map is used. Returns the mapped result.
  map_fn = _py_map
  # If the overridden function is not the same across all iterables, use _py_map
  for x in iterables:
    map_override = registry_lookup(map_registry, x)
    if map_override is None or (map_fn != _py_map and map_override != map_fn):  # pylint: disable=comparison-with-callable
      map_fn = _py_map
      break
    map_fn = map_override
  return map_fn(fn, *iterables)


def _py_map(fn, *iterables):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of map_.
  # Returns the Python lazy map of fn over the iterables.
  return map(fn, *iterables)


def next_(iterator, default=UNSPECIFIED):
  # [FUNCTION DOC]
  # Overload of next. Uses a registered override for the type of iterator (called with the
  # iterator and default) if any; otherwise the Python-level next_py. default is optional.
  # Returns the next item (or the default when exhausted and a default was given).
  next_override = registry_lookup(next_registry, iterator)
  if next_override is not None:
    return next_override(iterator, default)
  return next_py(iterator, default)


def next_py(iterator, default=UNSPECIFIED):
  # [FUNCTION DOC]
  # Plain-Python branch of next_ (public name, no underscore prefix). Advances the iterator and
  # returns the next item; if default was specified it is returned when the iterator is exhausted,
  # otherwise the stop-iteration error propagates. Side effect: consumes one item.
  if default is UNSPECIFIED:
    return next(iterator)
  return next(iterator, default)


def filter_(function, iterable):
  # [FUNCTION DOC]
  # Overload of filter. Uses a registered override for the type of iterable (called with the
  # predicate function and the iterable) if any; otherwise the Python filter. Returns the result.
  filter_override = registry_lookup(filter_registry, iterable)
  if filter_override is not None:
    return filter_override(function, iterable)
  return _py_filter(function, iterable)


def _py_filter(function, iterable):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of filter_.
  # Returns the Python lazy filter of iterable by the predicate function.
  return filter(function, iterable)


def any_(iterable):
  # [FUNCTION DOC]
  # Overload of any. Uses a registered override for the type of iterable if any; otherwise the
  # Python any. Returns a boolean (or whatever the override returns).
  any_override = registry_lookup(any_registry, iterable)
  if any_override is not None:
    return any_override(iterable)
  return _py_any(iterable)


def _py_any(iterable):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of any_.
  # Returns the Python any of iterable.
  return any(iterable)


def all_(iterable):
  # [FUNCTION DOC]
  # Overload of all. Uses a registered override for the type of iterable if any; otherwise the
  # Python all. Returns a boolean (or whatever the override returns).
  all_override = registry_lookup(all_registry, iterable)
  if all_override is not None:
    return all_override(iterable)
  return _py_all(iterable)


def _py_all(iterable):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of all_.
  # Returns the Python all of iterable.
  return all(iterable)


def sorted_(iterable, key=UNSPECIFIED, reverse=UNSPECIFIED):
  # [FUNCTION DOC]
  # Overload of sorted. key (sort-key function) and reverse (descending flag) are optional
  # (unspecified sentinel by default). Uses a registered override for the type of iterable (called
  # with iterable, key and reverse, sentinels included) if any; otherwise the Python sorted.
  # Returns the sorted list.
  sorted_override = registry_lookup(sorted_registry, iterable)
  if sorted_override is not None:
    return sorted_override(iterable, key, reverse)
  return _py_sorted(iterable, key, reverse)


def _py_sorted(iterable, key, reverse):
  # [FUNCTION DOC]
  # HELPER: plain-Python branch of sorted_.
  # Calls the Python sorted passing key and/or reverse only when they were specified (not the
  # sentinel). Returns the new sorted list.
  if key is not UNSPECIFIED and reverse is UNSPECIFIED:
    return sorted(iterable, key=key)
  if key is UNSPECIFIED and reverse is not UNSPECIFIED:
    return sorted(iterable, reverse=reverse)
  if key is not UNSPECIFIED and reverse is not UNSPECIFIED:
    return sorted(iterable, key=key, reverse=reverse)
  return sorted(iterable)


SUPPORTED_BUILTINS = (abs, float, int, len, print, range, enumerate, zip, map,
                      filter, any, all, sorted)

BUILTIN_FUNCTIONS_MAP = {
    'abs': abs_,
    'any': any_,
    'all': all_,
    'enumerate': enumerate_,
    'filter': filter_,
    'float': float_,
    'int': int_,
    'len': len_,
    'map': map_,
    'next': next_,
    'print': print_,
    'range': range_,
    'sorted': sorted_,
    'zip': zip_,
}
