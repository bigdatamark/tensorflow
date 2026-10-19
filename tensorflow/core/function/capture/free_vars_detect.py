# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Detects free variables (names a function uses but does not define) by static analysis of
# its source, following calls to functions defined in the same module, and builds a human
# readable log message about them. Intended only to warn users of compiled functions about
# possibly inconsistent behavior. Public entry point is the logging generator; the rest are
# private helpers. Defines a named record type for free variables and a module-level cache of
# generated messages (one per function) that is mutated at runtime.
# ==============================================================================
"""An independent module to detect free vars inside a function."""

import builtins
import collections
import functools
import inspect
import types

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity

FreeVar = collections.namedtuple("FreeVar", ["name", "is_function", "obj"])

_fn_log_cache = dict()


def _parse_and_analyze(func):
  # [FUNCTION DOC]
  # HELPER: static-analysis step for the free-variable search.
  # Parses the source of the function into a syntax tree, resolves qualified names, and runs the
  # activity (variable scope) analysis. Input: func, a Python function or lambda. Returns the
  # analyzed tree, whose scope annotation lists free variables. May raise a value error when
  # source is unavailable or a not-implemented error for ambiguous lambdas (callers catch these).
  """Parse and analyze Python Function code."""
  node, source = parser.parse_entity(func, future_features=())
  node = qual_names.resolve(node)
  entity_info = transformer.EntityInfo(
      name=func.__name__,
      source_code=source,
      source_file=None,
      future_features=(),
      namespace={})
  namer = naming.Namer({})
  ctx = transformer.Context(entity_info, namer, None)
  node = activity.resolve(node, ctx)
  return node


def _handle_wrap_partial_func(obj):
  # [FUNCTION DOC]
  # HELPER: unwraps decorated and partially applied callables for the free-variable search.
  # Repeatedly follows the wrapped-function attribute and the underlying function of partial
  # objects (functions and methods) until neither applies. Input: obj, any object. Returns the
  # innermost underlying object (the input itself if nothing to unwrap). No side effects.
  """Processes wrapped function and partial functions recursively."""
  modified = True
  while modified:
    modified = False
    while hasattr(obj, "__wrapped__"):
      obj = obj.__wrapped__
      modified = True
    if isinstance(obj, functools.partial) or isinstance(
        obj, functools.partialmethod):
      obj = obj.func
      modified = True
  return obj


def _get_self_obj_from_closure(fn):
  # [FUNCTION DOC]
  # HELPER: finds the object that the name self refers to inside a nested function.
  # Input: fn, a Python function that has a closure attribute (asserted). Splits the qualified
  # name of fn into parts, reversed (innermost first), and for each part scans the closure cells
  # (skipping empty cells) for a value whose type name equals that part. Returns the first
  # matching cell value, or null if none or the closure is empty.
  # KNOWN ISSUE: the check that the type is a class is always true for any value, and the match
  # uses the type name of the cell value, so any cell value whose type name appears in the
  # qualified name is returned, not necessarily self.
  """Get the object that `self` keyword refers to within a function.

  Args:
    fn: A python function object

  Returns:
    A class object that `self` refers to. Return None if not found.

  Here is an example demonstrating how this helper function works.

  ```
  class Foo():

    def __init__(self):
      self.val = 1

    def bar(self):
      x = 2

      def fn():
        return self.val + x

      return fn

  foo = Foo()
  fn = foo.bar()
  self_obj = _get_self_obj_from_closure(fn)
  assert self_obj is foo
  ```

  The goal is to get the `self_obj` (foo) from `fn`, so that it's feasible to
  access attributes of `foo`, like self.val in this case.

  This function first parses fn.qual_name, "Foo.bar.<locals>.fn", and finds the
  closure whose class name appear in fn.qual_name first.
  """
  assert hasattr(fn, "__closure__")
  qual_name = fn.__qualname__.split(".")
  # Search from the right to left
  qual_name = qual_name[::-1]

  if fn.__closure__:
    for cls_name in qual_name:
      for cell in fn.__closure__:
        try:
          closure = cell.cell_contents
        except ValueError:
          # Continue when cell is empty and its content is unavailable
          continue
        if inspect.isclass(type(closure)):
          if type(closure).__name__ == cls_name:
            obj = closure
            return obj

  return None


def _search_callable_free_vars(fn):
  # [FUNCTION DOC]
  # HELPER: lists the relevant free variables of one callable.
  # Input: fn, a Python function, method, lambda, partial or wrapped function (unwrapped first).
  # Returns a list of free-variable records sorted by name; empty if the source is unavailable
  # or the lambda is ambiguous. Skips builtins, subscripted names (only the base is kept),
  # modules and classes, and attribute chains unless rooted at self (then only the first
  # attribute is examined, resolved through the bound object for methods or through the closure
  # for plain functions). Function or method free variables are kept as function records only if
  # defined in the same module as fn (after unwrapping); other values become non-function records
  # with no object. Names that cannot be resolved are treated as plain values.
  """Search free vars from a callable object."""
  fn = _handle_wrap_partial_func(fn)

  try:
    node = _parse_and_analyze(fn)
  except ValueError:
    # When source code unavailable, return empty result
    return []
  except NotImplementedError:
    # Autograph cannot handle multiple lambda functions with same line number
    # and args name.
    return []

  scope = anno.getanno(node, anno.Static.SCOPE)
  free_vars_all = list(scope.free_vars)
  namespace = inspect_utils.getnamespace(fn)
  filtered = []

  for var in free_vars_all:
    base = str(var.qn[0])

    if var.is_simple():
      if base in builtins.__dict__.keys():
        continue
      obj = namespace.get(base, None)
    else:
      assert var.is_composite()
      # A compositve qualified name `QN` can be either an attr or a subscript
      if var.has_subscript():
        # For free var with subscripts, both the base and full formats are
        # generated.
        # For example, if the code have `glob[idx]`, `free_vars_all` would
        # contain `glob` as well as `glob[idx]`.
        # The method only keeps the base format for simplicity.
        continue
      else:
        assert var.has_attr()
        # For free vars with multiple attributes like `f.g.h`,
        # just as the subscripts, multiple free vars (QN) are generated:
        # ['f', 'f.g', 'f.g.h']
        # If `f` is `self`, only process the first attribute `f.g`.
        # Otherwise, only process `f`.
        if not var.qn[0].is_composite() and base == "self":
          attr = str(var.qn[1])
          # For method, access the object that `self` refers to via __self__
          if hasattr(fn, "__self__"):
            obj = getattr(fn.__self__, attr, None)
          # For function (not method) `self` usage under enclosing class scope
          elif hasattr(fn, "__closure__"):
            self_obj = _get_self_obj_from_closure(fn)
            if self_obj:
              obj = getattr(self_obj, attr, None)
            else:
              continue
          else:
            continue
        else:
          continue

    if (inspect.ismodule(obj) or inspect.isclass(obj)):
      continue
    elif inspect.isfunction(obj) or inspect.ismethod(obj):
      obj = _handle_wrap_partial_func(obj)
      if obj.__module__ != fn.__module__:
        continue
      filtered.append(FreeVar(str(var), True, obj))
    else:
      filtered.append(FreeVar(str(var), False, None))

  filtered = sorted(filtered, key=lambda x: x.name)
  return filtered


def _make_lambda_name(obj):
  # [FUNCTION DOC]
  # HELPER: derives a display name for a lambda. Reads the source text of obj and returns the
  # text before the first equals sign, trimmed (normally the variable the lambda is assigned
  # to). May raise an OS error if source is unavailable. If the source has no equals sign, the
  # whole trimmed source line is returned.
  source = inspect.getsource(obj)
  name = source.split("=")[0].strip()
  return name


def _make_callable_signature(obj):
  # [FUNCTION DOC]
  # HELPER: produces the string key used to identify a callable in the free-variable map.
  # Input: obj, a class, function or bound method. Classes and functions give their name (a
  # lambda gives its assigned name); methods give the owning class name, a dot, and the method
  # name (using the class itself when bound to a class). Raises a type error for any other kind
  # of object.
  """Generate signature for function/method."""
  if inspect.isclass(obj) or inspect.isfunction(obj):
    if obj.__name__ == "<lambda>":
      return _make_lambda_name(obj)
    return obj.__name__
  elif inspect.ismethod(obj):
    obj_self = obj.__self__
    if isinstance(obj_self, type):
      cls_name = obj_self.__name__
    else:
      cls_name = obj_self.__class__.__name__
    return f"{cls_name}.{obj.__name__}"
  else:
    raise TypeError(
        f"Only class/function/methods are valid inputs, got {type(obj)}")


def _detect_function_free_vars(fn):
  # [FUNCTION DOC]
  # HELPER: breadth-first search for free variables across fn and the same-module functions it
  # uses. Input: fn, a Python function or bound method (asserted, otherwise an assertion error
  # with the offending type). Returns a mapping from callable signature string to the sorted list
  # of its free-variable records; callables with no free variables are omitted. Each callable
  # is analyzed at most once per signature.
  """Detect free vars in any Python function."""
  assert isinstance(fn, types.FunctionType) or isinstance(
      fn, types.MethodType
  ), f"The input should be of Python function type. Got type: {type(fn)}."

  queue = collections.deque([fn])
  fn_map = dict()

  # Perform BFS over functions to get free vars
  while queue:
    obj = queue.popleft()
    signature = _make_callable_signature(obj)
    if signature not in fn_map:
      free_vars = _search_callable_free_vars(obj)
      if not free_vars:
        continue
      fn_map[signature] = free_vars
      for var in free_vars:
        # Only search callable free vars
        if var.is_function:
          obj = var.obj
          if _make_callable_signature(obj) not in fn_map:
            queue.append(obj)

  # func_name -> namedtupe FreeVar
  return fn_map


def generate_free_var_logging(fn, fn_threshold=5, var_threshold=10):
  # [FUNCTION DOC]
  # Public entry point: builds a warning message listing free variables found in fn.
  # Inputs: fn (function, method, partial or partial method; others give null), fn_threshold
  # (default 5; maximum number of functions shown including the top one) and var_threshold
  # (default 10; maximum variable names per function before an ellipsis).
  # Returns null if the input is unsupported, lacks module or qualified name, was already logged
  # (a module-level cache keyed by module and qualified name prevents repeats), or analysis
  # failed. Returns null and caches null if no free variables are found. Otherwise returns a
  # multi-line string: an explanation, a line for fn itself, lines for other functions sorted by
  # name, and a final ellipsis line if more functions exist; the string is cached.
  # Side effect: updates the module-level cache. Never raises for analysis errors.
  # KNOWN ISSUE: if fn itself has no free variables but other functions do, looking up fn's line
  # fails and an empty string is returned (and not cached). The explanation text also lacks
  # spaces between concatenated sentences ("in" + module, "code" + "if", "in" + URL).
  """Generate loggings of free vars from fn."""
  # Now only detect free vars for function/method
  if not (isinstance(fn, types.FunctionType) or isinstance(
      fn, types.MethodType) or isinstance(fn, functools.partial) or
          isinstance(fn, functools.partialmethod)):
    return None
  fn = _handle_wrap_partial_func(fn)

  if not (hasattr(fn, "__module__") and hasattr(fn, "__qualname__")):
    return None
  fn_key = (fn.__module__, fn.__qualname__)
  # To prevent log spam, only generate logging once for each tf.function
  if fn_key in _fn_log_cache:
    return None

  try:
    fn_vars_map = _detect_function_free_vars(fn)
  except Exception:  # pylint: disable=broad-except
    # Only for logging purpose, do not raise errors to users
    return None

  # If not free vars detected, return None
  if not fn_vars_map:
    _fn_log_cache[fn_key] = None
    return _fn_log_cache[fn_key]

  logging_txt = []
  tf_fn_name = _make_callable_signature(fn)
  tf_fn_module = fn.__module__

  def one_line_logging(fn_name, free_vars, threshold=10):
    # [FUNCTION DOC]
    # HELPER: formats one log line for a function. Inputs: fn_name (display string), free_vars
    # (list of free-variable records), threshold (default 10; maximum names listed). Returns an
    # empty string if there are no variables; otherwise a line naming the function followed by
    # comma-separated variable names, with a trailing ellipsis if the list exceeded threshold.
    if not free_vars:
      return ""
    log = f"Inside function {fn_name}(): "
    log += ", ".join([var.name for var in free_vars[:threshold]])
    if len(free_vars) > threshold:
      log += "..."
    return log

  # Show the free vars info of the tf.function at the top
  fn_threshold -= 1
  try:
    tf_fn_line = one_line_logging(tf_fn_name, fn_vars_map[tf_fn_name],
                                  var_threshold)
  except Exception:  # pylint: disable=broad-except
    # Only for logging purpose, do not raise error to users
    return ""

  # Functions that are defined outside of tf.function
  outside_fn_lines = []
  outside_fn_names = [name for name in fn_vars_map.keys() if name != tf_fn_name]
  outside_fn_names = sorted(outside_fn_names)
  for fn_name in outside_fn_names[:fn_threshold]:
    outside_fn_lines.append(
        one_line_logging(fn_name, fn_vars_map[fn_name], var_threshold))

  if len(fn_vars_map) > fn_threshold:
    ellipsis_line = "..."
  else:
    ellipsis_line = None

  # TODO(panzf): direct users to the manual API after it's exposed to public
  explanation_line = (
      f"Free variables are detected within tf.function {tf_fn_name}() in"
      f"{tf_fn_module}. Free variable usage may cause inconsistant behaviors"
      "between eager mode and tf.function. Please consider refactor the code"
      "if possible. More details are avaiable in"
      "https://www.tensorflow.org/guide/function#limitations.\n"
      "Free variable names inside each function/method are shown below:")

  logging_txt = [explanation_line, tf_fn_line] + outside_fn_lines
  if ellipsis_line:
    logging_txt.append(ellipsis_line)
  logging_txt = "\n".join(logging_txt)

  _fn_log_cache[fn_key] = logging_txt
  return _fn_log_cache[fn_key]
