# [MODULE DOC]
# Implements the "should use result" facility: a decorator that wraps the return value of a
# function in a thin proxy object which remembers whether anything was done with it (attribute
# read or write, operator, or an explicit mark-used call). If the proxy is destroyed without
# having been used, an error is logged (or, for values created during function tracing when
# requested, a runtime error is raised when tracing ends), pointing at the creation stack.
# Exposes the public decorator should_use_result and the internal wrapping function
# add-should-use-warning, plus the usage-tracking helper class and the proxy-building
# machinery. Keeps a module-level cache of generated proxy classes keyed by wrapped type.
# Used by TensorFlow code that returns ops or tensors that callers must not silently drop.
# No import-time side effects other than creating the empty cache and the operator name set.
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
"""Decorator that provides a warning if the wrapped object is never used."""
import copy
import sys
import textwrap
import traceback
import types

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator


class _TFShouldUseHelper(object):
  # [CLASS DOC]
  # HELPER: usage tracker held by the proxy objects created for the should-use-result feature.
  # State: the wrapped value's type, its text representation, the stack frame of the creator
  # (for the report), the flag saying whether to raise inside functions, and a "sated" flag
  # that becomes true once the wrapped value has been used. When the helper is garbage
  # collected while not sated it logs an error naming the creation stack. In function-tracing
  # mode with error requested it also registers an exit callback on the default function graph
  # that raises a runtime error if the value is still unused when tracing finishes.
  """Object stored in TFShouldUse-wrapped objects.

  When it is deleted it will emit a warning or error if its `sate` method
  has not been called by time of deletion, and Tensorflow is not executing
  eagerly or inside a tf.function (which use autodeps and resolve the
  main issues this wrapper warns about).
  """

  def __init__(self, type_, repr_, stack_frame, error_in_function,
               warn_in_eager):
    # [FUNCTION DOC]
    # Constructor. Inputs: type_ (type of the wrapped value), repr_ (its text form), stack_frame
    # (frame object of the creator, used later to format a stack), error_in_function (bool:
    # raise instead of log when created while tracing a function) and warn_in_eager (bool:
    # also track in eager mode). Decides the initial sated state: in eager mode it is sated
    # unless warn_in_eager is set; inside a function it is unsated only if error_in_function
    # (and then an exit callback raising on unused value is registered on the default function
    # graph; otherwise it is sated); in classic graph-building mode it starts unsated.
    # Returns nothing; side effect is the callback registration described.
    self._type = type_
    self._repr = repr_
    self._stack_frame = stack_frame
    self._error_in_function = error_in_function
    if context.executing_eagerly():
      # If warn_in_eager, sated == False.  Otherwise true.
      self._sated = not warn_in_eager
    elif ops.inside_function():
      if error_in_function:
        self._sated = False
        ops.add_exit_callback_to_default_func_graph(
            lambda: self._check_sated(raise_error=True))
      else:
        self._sated = True
    else:
      # TF1 graph building mode
      self._sated = False

  def sate(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: also assigns an attribute named logging module that is never read anywhere
    # else in this class; it appears to be a leftover.
    # Marks the tracked value as used: sets the sated flag and drops the stored type, text
    # representation and creation frame so they can be freed. Takes no inputs, returns nothing.
    self._sated = True
    self._type = None
    self._repr = None
    self._stack_frame = None
    self._logging_module = None

  def _check_sated(self, raise_error):
    # [FUNCTION DOC]
    # HELPER: internal check shared by the destructor and the function-graph exit callback.
    # Input: raise_error (bool). If the value was already used, returns nothing immediately.
    # Otherwise builds a text of up to five frames of the creation stack. If raise_error is
    # true it raises a runtime error saying the object was never used (with its type, text
    # form and creation stack) and, in all cases of that branch, marks the helper sated
    # afterwards. If false, it logs the same information as an error message and does not
    # change state.
    """Check if the object has been sated."""
    if self._sated:
      return
    creation_stack = ''.join(
        [line.rstrip()
         for line in traceback.format_stack(self._stack_frame, limit=5)])
    if raise_error:
      try:
        raise RuntimeError(
            'Object was never used (type {}): {}.  If you want to mark it as '
            'used call its "mark_used()" method.  It was originally created '
            'here:\n{}'.format(self._type, self._repr, creation_stack))
      finally:
        self.sate()
    else:
      tf_logging.error(
          '==================================\n'
          'Object was never used (type {}):\n{}\nIf you want to mark it as '
          'used call its "mark_used()" method.\nIt was originally created '
          'here:\n{}\n'
          '=================================='
          .format(self._type, self._repr, creation_stack))

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor. When the helper is collected it runs the sated check in logging mode, so an
    # unused tracked value results in an error log (never an exception). Returns nothing.
    self._check_sated(raise_error=False)


def _new__init__(self, wrapped_value, tf_should_use_helper):
  # [FUNCTION DOC]
  # HELPER: installed as the constructor of the generated proxy classes.
  # Stores the usage tracker and the wrapped value on the proxy under private attribute names.
  # Inputs: the proxy instance, the value being wrapped and the tracker object. Returns nothing.
  # pylint: disable=protected-access
  self._tf_should_use_helper = tf_should_use_helper
  self._tf_should_use_wrapped_value = wrapped_value


def _new__setattr__(self, key, value):
  # [FUNCTION DOC]
  # HELPER: installed as the attribute-setting hook of the generated proxy classes.
  # Inputs: proxy instance, attribute name and value. The two private bookkeeping attributes are
  # stored on the proxy itself; any other attribute is forwarded to the wrapped value. Note that
  # setting forwards without marking the tracker as used here (the attribute lookup used to
  # fetch the wrapped value is the private one, which does not mark use). Returns whatever the
  # underlying set operation returns (nothing in practice).
  if key in ('_tf_should_use_helper', '_tf_should_use_wrapped_value'):
    return object.__setattr__(self, key, value)
  return setattr(
      object.__getattribute__(self, '_tf_should_use_wrapped_value'),
      key, value)


def _new__getattribute__(self, key):
  # [FUNCTION DOC]
  # HELPER: installed as the attribute-access hook of the generated proxy classes.
  # Inputs: proxy instance and attribute name. Unless the name is one of the two private
  # bookkeeping attributes, marks the tracker as used. Then returns the proxy's own attribute
  # for the bookkeeping names, the mark-used method and the attribute-setting hook; for every
  # other name returns the attribute of the wrapped value. Raises whatever attribute error the
  # wrapped value raises for unknown names.
  if key not in ('_tf_should_use_helper', '_tf_should_use_wrapped_value'):
    object.__getattribute__(self, '_tf_should_use_helper').sate()
  if key in (
      '_tf_should_use_wrapped_value',
      '_tf_should_use_helper',
      'mark_used',
      '__setattr__',
  ):
    return object.__getattribute__(self, key)
  return getattr(
      object.__getattribute__(self, '_tf_should_use_wrapped_value'), key)


def _new_mark_used(self, *args, **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the attribute-error handler wraps the whole call, so an attribute error raised
  # inside the wrapped value's own mark-used method is silently swallowed and the result is
  # nothing, not only the "method does not exist" case.
  # HELPER: installed as the mark-used method of the generated proxy classes.
  # Marks the tracker as used, then if the wrapped value itself has a mark-used method calls it
  # with the same positional and keyword arguments and returns its result. If the wrapped value
  # has none, returns nothing.
  object.__getattribute__(self, '_tf_should_use_helper').sate()
  try:
    mu = object.__getattribute__(
        object.__getattribute__(self, '_tf_should_use_wrapped_value'),
        'mark_used')
    return mu(*args, **kwargs)
  except AttributeError:
    pass

OVERLOADABLE_OPERATORS = {
    '__add__',
    '__radd__',
    '__sub__',
    '__rsub__',
    '__mul__',
    '__rmul__',
    '__div__',
    '__rdiv__',
    '__truediv__',
    '__rtruediv__',
    '__floordiv__',
    '__rfloordiv__',
    '__mod__',
    '__rmod__',
    '__lt__',
    '__le__',
    '__gt__',
    '__ge__',
    '__ne__',
    '__eq__',
    '__and__',
    '__rand__',
    '__or__',
    '__ror__',
    '__xor__',
    '__rxor__',
    '__getitem__',
    '__pow__',
    '__rpow__',
    '__invert__',
    '__neg__',
    '__abs__',
    '__matmul__',
    '__rmatmul__',
}


_WRAPPERS = {}


class ShouldUseWrapper(object):
  # [CLASS DOC]
  # HELPER: empty template class. It is deep-copied and rebuilt by the proxy factory, which then
  # replaces its constructor, attribute hooks and mark-used method and copies operator methods
  # from the wrapped value's type. It is never instantiated directly and holds no state.
  pass


def _get_wrapper(x, tf_should_use_helper):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the wrapper class subclasses the type of x, but the
  # generated class is built from the bases of the empty template class (plain object) and only
  # copies selected operator methods from the type of x, so isinstance checks against the type
  # of x fail on the proxy.
  # HELPER: proxy factory for add-should-use-warning.
  # Inputs: x (value to wrap) and the usage tracker. Looks up a cached proxy class for the type
  # of x in the module-level cache; if none, creates one by copying the template class,
  # installing the custom constructor, attribute-access hook, attribute-set hook and mark-used
  # method, copying every overloadable operator method that the type of x defines, and storing
  # it in the cache (side effect on module state). Returns a new proxy instance wrapping x.
  """Create a wrapper for object x, whose class subclasses type(x).

  The wrapper will emit a warning if it is deleted without any of its
  properties being accessed or methods being called.

  Args:
    x: The instance to wrap.
    tf_should_use_helper: The object that tracks usage.

  Returns:
    An object wrapping `x`, of type `type(x)`.
  """
  type_x = type(x)
  memoized = _WRAPPERS.get(type_x, None)
  if memoized:
    return memoized(x, tf_should_use_helper)

  # Make a copy of `object`
  tx = copy.deepcopy(ShouldUseWrapper)
  # Prefer using __orig_bases__, which preserve generic type arguments.
  bases = getattr(tx, '__orig_bases__', tx.__bases__)

  def set_body(ns):
    # [FUNCTION DOC]
    # HELPER: class-body callback for dynamic class creation. Fills the namespace mapping given
    # by the class machinery with the attributes of the copied template class and returns it.
    ns.update(tx.__dict__)
    return ns

  copy_tx = types.new_class(tx.__name__, bases, exec_body=set_body)
  copy_tx.__init__ = _new__init__
  copy_tx.__getattribute__ = _new__getattribute__
  for op in OVERLOADABLE_OPERATORS:
    if hasattr(type_x, op):
      setattr(copy_tx, op, getattr(type_x, op))

  copy_tx.mark_used = _new_mark_used
  copy_tx.__setattr__ = _new__setattr__
  _WRAPPERS[type_x] = copy_tx

  return copy_tx(x, tf_should_use_helper)


def _add_should_use_warning(x, error_in_function=False, warn_in_eager=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring refers to a class named TFShouldUseWarningWrapper that does not
  # exist in this module, and says the result subclasses the type of x, which is not true (see
  # the proxy factory).
  # Wraps a value so that an unused value is reported. Inputs: x (any value), error_in_function
  # (bool, default false: raise at the end of function tracing if unused) and warn_in_eager
  # (bool, default false: also track in eager mode). Returns x unchanged when it is nothing, an
  # empty list, when executing eagerly without warn_in_eager, or when inside a function without
  # error_in_function. Otherwise captures the caller's stack frame (by raising and catching an
  # exception to read its traceback), creates the usage tracker and returns a proxy wrapping x.
  # Side effects: possible registration of a function-graph exit callback via the tracker.
  """Wraps object x so that if it is never used, a warning is logged.

  Args:
    x: Python object.
    error_in_function: Python bool.  If `True`, a `RuntimeError` is raised
      if the returned value is never used when created during `tf.function`
      tracing.
    warn_in_eager: Python bool. If `True` raise warning if in Eager mode as well
      as graph mode.

  Returns:
    An instance of `TFShouldUseWarningWrapper` which subclasses `type(x)`
    and is a very shallow wrapper for `x` which logs access into `x`.
  """
  if x is None or (isinstance(x, list) and not x):
    return x

  if context.executing_eagerly() and not warn_in_eager:
    return x

  if ops.inside_function() and not error_in_function:
    # We don't currently log warnings in tf.function calls, so just skip it.
    return x

  # Extract the current frame for later use by traceback printing.
  try:
    raise ValueError()
  except ValueError:
    stack_frame = sys.exc_info()[2].tb_frame.f_back

  tf_should_use_helper = _TFShouldUseHelper(
      type_=type(x),
      repr_=repr(x),
      stack_frame=stack_frame,
      error_in_function=error_in_function,
      warn_in_eager=warn_in_eager)

  return _get_wrapper(x, tf_should_use_helper)


def should_use_result(fn=None, warn_in_eager=False, error_in_function=False):
  # [FUNCTION DOC]
  # Public decorator factory. Inputs: fn (optional function to decorate directly), warn_in_eager
  # and error_in_function (bools, default false, forwarded to the wrapping logic). If fn is
  # given, returns the decorated function; if not, returns a decorator to apply later. The
  # decorated function calls the original and returns its result passed through the
  # add-should-use-warning logic, and its documentation is the original brief line plus
  # dedented remainder followed by a note that the output must be used. Decoration goes through
  # the tf decorator maker so the original is recorded as the target.
  """Function wrapper that ensures the function's output is used.

  If the output is not used, a `logging.error` is logged.  If
  `error_in_function` is set, then a `RuntimeError` will be raised at the
  end of function tracing if the output is not used by that point.

  An output is marked as used if any of its attributes are read, modified, or
  updated.  Examples when the output is a `Tensor` include:

  - Using it in any capacity (e.g. `y = t + 0`, `sess.run(t)`)
  - Accessing a property (e.g. getting `t.name` or `t.op`).
  - Calling `t.mark_used()`.

  Note, certain behaviors cannot be tracked - for these the object may not
  be marked as used.  Examples include:

  - `t != 0`.  In this case, comparison is done on types / ids.
  - `isinstance(t, tf.Tensor)`.  Similar to above.

  Args:
    fn: The function to wrap.
    warn_in_eager: Whether to create warnings in Eager as well.
    error_in_function: Whether to raise an error when creating a tf.function.

  Returns:
    The wrapped function.
  """
  def decorated(fn):
    # [FUNCTION DOC]
    # HELPER: actual decorator used by the public factory. Input: fn, the function to wrap.
    # Builds the wrapper, composes the new documentation (brief line, dedented rest, plus a
    # note that the output should be used and can be marked via the mark-used method) and
    # returns the result of the tf decorator maker named should-use-result.
    """Decorates the input function."""
    def wrapped(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper installed by the decorator. Calls the original function with all given
      # arguments and returns its result wrapped by the unused-value tracking logic using the
      # configured warn-in-eager and error-in-function flags. Propagates exceptions of the call.
      return _add_should_use_warning(fn(*args, **kwargs),
                                     warn_in_eager=warn_in_eager,
                                     error_in_function=error_in_function)
    fn_doc = fn.__doc__ or ''
    split_doc = fn_doc.split('\n', 1)
    if len(split_doc) == 1:
      updated_doc = fn_doc
    else:
      brief, rest = split_doc
      updated_doc = '\n'.join([brief, textwrap.dedent(rest)])

    note = ('\n\nNote: The output of this function should be used. If it is '
            'not, a warning will be logged or an error may be raised. '
            'To mark the output as used, call its .mark_used() method.')
    return tf_decorator.make_decorator(
        target=fn,
        decorator_func=wrapped,
        decorator_name='should_use_result',
        decorator_doc=updated_doc + note)

  if fn is not None:
    return decorated(fn)
  else:
    return decorated
