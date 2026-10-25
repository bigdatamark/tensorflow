# [MODULE DOC]
# Core engine of the TensorFlow Python upgrade scripts. Given an API change specification (a
# class with tables of symbol renames, keyword renames, argument reorders, warnings, custom
# transformers, module deprecations and import renames) it parses Python source with the pasta
# library (which preserves formatting), walks the syntax tree with a visitor that applies the
# specification, and writes the modified source plus a log of changes and a list of warnings and
# errors. Exposes: severity constants, an import-rename record type, tree-building and
# argument-inspection helpers, the specification base classes, an edit visitor, an analysis
# visitor that only detects usage, and the code upgrader that processes strings, files and
# directory trees. Import-time side effects: compiles two regular expressions and defines
# constants; requires the pasta package. Used by the version-specific upgrade scripts.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Upgrader for Python scripts according to an API change specification."""

import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback

import pasta


# Some regular expressions we will need for parsing
FIND_OPEN = re.compile(r"^\s*(\[).*$")
FIND_STRING_CHARS = re.compile(r"['\"]")


INFO = "INFO"
WARNING = "WARNING"
ERROR = "ERROR"


ImportRename = collections.namedtuple(
    "ImportRename", ["new_name", "excluded_prefixes"])


def full_name_node(name, ctx=ast.Load()):
  # [FUNCTION DOC]
  # HELPER: Builds syntax-tree nodes for qualified names used by rename logic.
  # KNOWN ISSUE: the default context is a single shared instance created at definition time,
  # and the returned outermost node's context is set to the given context object itself, so
  # callers sharing the default share one context object. Behaviorally harmless in practice.
  # Inputs: name, a dotted string such as a.b.c; ctx, the expression context to assign to the
  # outermost node (default: load context). Output: a name node for a single component, or a
  # chain of attribute nodes nested left to right for dotted names; inner nodes always get load
  # context. Raises nothing for normal strings. No side effects.
  """Make an Attribute or Name node for name.

  Translate a qualified name into nested Attribute nodes (and a Name node).

  Args:
    name: The name to translate to a node.
    ctx: What context this name is used in. Defaults to Load()

  Returns:
    A Name or Attribute node.
  """
  names = name.split(".")
  names.reverse()
  node = ast.Name(id=names.pop(), ctx=ast.Load())
  while names:
    node = ast.Attribute(value=node, attr=names.pop(), ctx=ast.Load())

  # Change outermost ctx to the one given to us (inner ones should be Load).
  node.ctx = ctx
  return node


def get_arg_value(node, arg_name, arg_pos=None):
  # [FUNCTION DOC]
  # HELPER: Argument lookup used by call-warning and transformer logic.
  # KNOWN ISSUE: when a starred positional argument precedes the wanted position, it is skipped
  # without advancing the index, so the positional lookup may return an argument that is not
  # actually at that position at run time.
  # Inputs: node, a call node; arg_name, keyword name to look for (may be nothing/null to skip
  # the keyword search); arg_pos, optional zero-based position to check among positional
  # arguments. Keywords are searched first, then positional arguments with starred ones skipped.
  # Output: a pair of a boolean telling whether the argument was found and the value node (or
  # nothing/null). No side effects, no exceptions.
  """Get the value of an argument from a ast.Call node.

  This function goes through the positional and keyword arguments to check
  whether a given argument was used, and if so, returns its value (the node
  representing its value).

  This cannot introspect *args or **args, but it safely handles *args in
  Python3.5+.

  Args:
    node: The ast.Call node to extract arg values from.
    arg_name: The name of the argument to extract.
    arg_pos: The position of the argument (in case it's passed as a positional
      argument).

  Returns:
    A tuple (arg_present, arg_value) containing a boolean indicating whether
    the argument is present, and its value in case it is.
  """
  # Check keyword args
  if arg_name is not None:
    for kw in node.keywords:
      if kw.arg == arg_name:
        return (True, kw.value)

  # Check positional args
  if arg_pos is not None:
    idx = 0
    for arg in node.args:
      if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
        continue  # Can't parse Starred
      if idx == arg_pos:
        return (True, arg)
      idx += 1

  return (False, None)


def uses_star_args_in_call(node):
  # [FUNCTION DOC]
  # HELPER: Detects variadic positional arguments in a call.
  # KNOWN ISSUE: the docstring's Returns text mentions keyword args too, but only starred
  # positional arguments are checked. The legacy branch for Python older than 3.5 is dead on
  # any supported interpreter.
  # Input: node, a call node. Output: true if any positional argument is a starred expression
  # (or, on very old Pythons, the node has a star-args field), else false. No side effects.
  """Check if an ast.Call node uses arbitrary-length positional *args.

  This function works with the AST call node format of Python3.5+
  as well as the different AST format of earlier versions of Python.

  Args:
    node: The ast.Call node to check arg values for.

  Returns:
    True if the node uses starred variadic positional args or keyword args.
    False if it does not.
  """
  if sys.version_info[:2] >= (3, 5):
    # Check for an *args usage in python 3.5+
    for arg in node.args:
      if isinstance(arg, ast.Starred):
        return True
  else:
    if node.starargs:
      return True
  return False


def uses_star_kwargs_in_call(node):
  # [FUNCTION DOC]
  # HELPER: Detects variadic keyword arguments in a call.
  # KNOWN ISSUE: the docstring's Returns text mentions positional args too, but only
  # double-star keyword expansion is checked. The pre-3.5 branch is dead on supported Pythons.
  # Input: node, a call node. Output: true if any keyword entry has no name (a double-star
  # expansion), else false. No side effects.
  """Check if an ast.Call node uses arbitrary-length **kwargs.

  This function works with the AST call node format of Python3.5+
  as well as the different AST format of earlier versions of Python.

  Args:
    node: The ast.Call node to check arg values for.

  Returns:
    True if the node uses starred variadic positional args or keyword args.
    False if it does not.
  """
  if sys.version_info[:2] >= (3, 5):
    # Check for a **kwarg usage in python 3.5+
    for keyword in node.keywords:
      if keyword.arg is None:
        return True
  else:
    if node.kwargs:
      return True
  return False


def uses_star_args_or_kwargs_in_call(node):
  # [FUNCTION DOC]
  # HELPER: Combined variadic-argument check used by call visitors.
  # Input: node, a call node. Output: true if the call uses starred positional arguments or
  # double-star keyword expansion, else false. No side effects.
  """Check if an ast.Call node uses arbitrary-length *args or **kwargs.

  This function works with the AST call node format of Python3.5+
  as well as the different AST format of earlier versions of Python.

  Args:
    node: The ast.Call node to check arg values for.

  Returns:
    True if the node uses starred variadic positional args or keyword args.
    False if it does not.
  """
  return uses_star_args_in_call(node) or uses_star_kwargs_in_call(node)


def excluded_from_module_rename(module, import_rename_spec):
  # [FUNCTION DOC]
  # HELPER: Exclusion test used when renaming imports.
  # Inputs: module, a dotted module name string; import_rename_spec, an import-rename record
  # holding a new name and a collection of excluded name prefixes. Output: true if the module
  # name starts with any excluded prefix (plain string prefix, not component-aware), else
  # false. No side effects.
  """Check if this module import should not be renamed.

  Args:
    module: (string) module name.
    import_rename_spec: ImportRename instance.

  Returns:
    True if this import should not be renamed according to the
    import_rename_spec.
  """
  for excluded_prefix in import_rename_spec.excluded_prefixes:
    if module.startswith(excluded_prefix):
      return True
  return False


class APIChangeSpec:
  # [CLASS DOC]
  # Base class for API change specifications. Concrete subclasses (for example the TF 2.0 spec)
  # supply the rule tables listed in the docstring as attributes; this base supplies only
  # no-op preprocessing hooks. KNOWN ISSUE: the base class defines none of the documented
  # fields itself, and the visitor also reads optional attributes not listed in the docstring
  # (function arg warnings, max submodule depth, inserts after imports).
  """This class defines the transformations that need to happen.

  This class must provide the following fields:

  * `function_keyword_renames`: maps function names to a map of old -> new
    argument names
  * `symbol_renames`: maps function names to new function names
  * `change_to_function`: a set of function names that have changed (for
    notifications)
  * `function_reorders`: maps functions whose argument order has changed to the
    list of arguments in the new order
  * `function_warnings`: maps full names of functions to warnings that will be
    printed out if the function is used. (e.g. tf.nn.convolution())
  * `function_transformers`: maps function names to custom handlers
  * `module_deprecations`: maps module names to warnings that will be printed
    if the module is still used after all other transformations have run
  * `import_renames`: maps import name (must be a short name without '.')
    to ImportRename instance.

  For an example, see `TFAPIChangeSpec`.
  """

  def preprocess(self, root_node):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Hook called on a freshly parsed tree before visiting. Input: root_node, the tree root.
    # Default behavior: returns the same root plus an empty list of log entries and an empty
    # list of errors. Subclasses may rewrite the tree and add rules (undone by the clearing hook).
    """Preprocess a parse tree. Return a preprocessed node, logs and errors."""
    return root_node, [], []

  def clear_preprocessing(self):
    # [FUNCTION DOC]
    # Hook called after a file has been visited to undo rule changes made during preprocessing.
    # Default: does nothing, returns nothing.
    """Restore this APIChangeSpec to before it preprocessed a file.

    This is needed if preprocessing a file changed any rewriting rules.
    """
    pass


class NoUpdateSpec(APIChangeSpec):
  # [CLASS DOC]
  # HELPER: Empty specification used by the analysis visitor so that inherited edit machinery
  # has all rule tables present but empty. All tables are plain empty mappings created per
  # instance.
  """A specification of an API change which doesn't change anything."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no inputs. Creates empty mappings for function handles, reorders,
    # keyword renames, symbol renames, warnings, change-to-function, module deprecations,
    # function transformers and import renames. Returns nothing.
    self.function_handle = {}
    self.function_reorders = {}
    self.function_keyword_renames = {}
    self.symbol_renames = {}
    self.function_warnings = {}
    self.change_to_function = {}
    self.module_deprecations = {}
    self.function_transformers = {}
    self.import_renames = {}


class _PastaEditVisitor(ast.NodeVisitor):
  # [CLASS DOC]
  # HELPER: Internal tree visitor behind the code upgrader.
  # Applies a change specification to a pasta-parsed tree: renames symbols, makes positional
  # arguments keywords, renames or removes keyword arguments, runs custom transformers, adds
  # warnings, and rewrites imports. State: the specification, a log list of (severity, line,
  # column, message) tuples, and a stack of ancestor nodes so handlers can reach parents.
  # Subclassed by the analysis visitor.
  """AST Visitor that processes function calls.

  Updates function calls from old API version to new API version using a given
  change spec.
  """

  def __init__(self, api_change_spec):
    # [FUNCTION DOC]
    # Constructor. Input: api_change_spec, the specification object whose tables drive edits.
    # Stores it, and starts with an empty log list and an empty ancestor stack. Returns nothing.
    self._api_change_spec = api_change_spec
    self._log = []   # Holds 4-tuples: severity, line, col, msg.
    self._stack = []  # Allow easy access to parents.

  # Overridden to maintain a stack of nodes to allow for parent access
  def visit(self, node):
    # [FUNCTION DOC]
    # Override of the generic visit entry. Input: node, the tree node to visit. Pushes the node
    # on the ancestor stack, dispatches to the normal visitor, then pops it (not exception safe:
    # if a handler raises, the node stays on the stack). Returns nothing.
    self._stack.append(node)
    super(_PastaEditVisitor, self).visit(node)
    self._stack.pop()

  @property
  def errors(self):
    # [FUNCTION DOC]
    # Property returning a new list of log entries whose severity is error.
    return [log for log in self._log if log[0] == ERROR]

  @property
  def warnings(self):
    # [FUNCTION DOC]
    # Property returning a new list of log entries whose severity is warning.
    return [log for log in self._log if log[0] == WARNING]

  @property
  def warnings_and_errors(self):
    # [FUNCTION DOC]
    # Property returning a new list of log entries whose severity is warning or error, in log
    # order. Used to build the per-file error report.
    return [log for log in self._log if log[0] in (WARNING, ERROR)]

  @property
  def info(self):
    # [FUNCTION DOC]
    # Property returning a new list of log entries whose severity is informational.
    return [log for log in self._log if log[0] == INFO]

  @property
  def log(self):
    # [FUNCTION DOC]
    # Property returning the live list of all log entries (severity, line, column, message).
    return self._log

  def add_log(self, severity, lineno, col, msg):
    # [FUNCTION DOC]
    # Records one log entry. Inputs: severity string, line number, column offset, message text.
    # Side effects: appends the tuple to the log and prints a formatted line to standard output.
    # Returns nothing.
    self._log.append((severity, lineno, col, msg))
    print("%s line %d:%d: %s" % (severity, lineno, col, msg))

  def add_logs(self, logs):
    # [FUNCTION DOC]
    # Records many log entries at once. Input: logs, a list of (severity, line, column,
    # message) tuples, typically produced by custom transformers. Side effects: extends the log
    # and prints each entry to standard output. Raises a formatting error if an entry is not a
    # four-element tuple of the expected types. Returns nothing.
    """Record a log and print it.

    The log should be a tuple `(severity, lineno, col_offset, msg)`, which will
    be printed and recorded. It is part of the log available in the `self.log`
    property.

    Args:
      logs: The logs to add. Must be a list of tuples
        `(severity, lineno, col_offset, msg)`.
    """
    self._log.extend(logs)
    for log in logs:
      print("%s line %d:%d: %s" % log)

  def _get_applicable_entries(self, transformer_field, full_name, name):
    # [FUNCTION DOC]
    # HELPER: Rule lookup for call handling.
    # Inputs: transformer_field, name of a specification attribute holding a mapping (missing
    # attribute treated as empty); full_name, dotted callable name or nothing; name, short name
    # or nothing. Output: a list (possibly empty, in order) of the entries stored under the full
    # name, under the wildcard-dot-name key, and under the single wildcard key. No side effects.
    """Get all list entries indexed by name that apply to full_name or name."""
    # Transformers are indexed to full name, name, or no name
    # as a performance optimization.
    function_transformers = getattr(self._api_change_spec,
                                    transformer_field, {})

    glob_name = "*." + name if name else None
    transformers = []
    if full_name in function_transformers:
      transformers.append(function_transformers[full_name])
    if glob_name in function_transformers:
      transformers.append(function_transformers[glob_name])
    if "*" in function_transformers:
      transformers.append(function_transformers["*"])
    return transformers

  def _get_applicable_dict(self, transformer_field, full_name, name):
    # [FUNCTION DOC]
    # HELPER: Rule lookup for keyword renames and argument warnings.
    # Inputs: as for the entries lookup. Output: a new mapping merging a copy of the wildcard
    # entry, then the wildcard-dot-name entry, then the full-name entry, with later (more
    # specific) ones overriding earlier keys. Missing entries count as empty. The specification
    # tables are not modified.
    """Get all dict entries indexed by name that apply to full_name or name."""
    # Transformers are indexed to full name, name, or no name
    # as a performance optimization.
    function_transformers = getattr(self._api_change_spec,
                                    transformer_field, {})

    glob_name = "*." + name if name else None
    transformers = function_transformers.get("*", {}).copy()
    transformers.update(function_transformers.get(glob_name, {}))
    transformers.update(function_transformers.get(full_name, {}))
    return transformers

  def _get_full_name(self, node):
    # [FUNCTION DOC]
    # HELPER: Name extraction used by all visit handlers.
    # Input: node, an attribute or name node (or any expression). Output: the dotted name made
    # from walking attribute chains down to a plain name, or nothing/null if the chain bottoms
    # out in something else such as a call or subscript. No side effects.
    """Traverse an Attribute node to generate a full name, e.g., "tf.foo.bar".

    This is the inverse of `full_name_node`.

    Args:
      node: A Node of type Attribute.

    Returns:
      a '.'-delimited full-name or None if node was not Attribute or Name.
      i.e. `foo()+b).bar` returns None, while `a.b.c` would return "a.b.c".
    """
    curr = node
    items = []
    while not isinstance(curr, ast.Name):
      if not isinstance(curr, ast.Attribute):
        return None
      items.append(curr.attr)
      curr = curr.value
    items.append(curr.id)
    return ".".join(reversed(items))

  def _maybe_add_warning(self, node, full_name):
    # [FUNCTION DOC]
    # HELPER: Emits a manual-check message for a symbol listed in the function warnings table.
    # Inputs: node, node providing line and column; full_name, dotted name (may be a wildcard
    # key built by the caller). If found, the stored level and message are retrieved, the
    # function-name placeholder is replaced by the name, and a log entry is added (printed).
    # Output: true if a warning was recorded, else false.
    """Adds an error to be printed about full_name at node."""
    function_warnings = self._api_change_spec.function_warnings
    if full_name in function_warnings:
      level, message = function_warnings[full_name]
      message = message.replace("<function name>", full_name)
      self.add_log(level, node.lineno, node.col_offset,
                   "%s requires manual check. %s" % (full_name, message))
      return True
    else:
      return False

  def _maybe_add_module_deprecation_warning(self, node, full_name, whole_name):
    # [FUNCTION DOC]
    # HELPER: Emits a warning when a member of a deprecated module is used.
    # Inputs: node, for position; full_name, dotted module name looked up in the module
    # deprecations table; whole_name, the dumped source text of the full expression using it.
    # If found, adds a log entry naming the member and module with the stored message (placeholder
    # replaced by the whole name). Output: true if a warning was added, else false.
    """Adds a warning if full_name is a deprecated module."""
    warnings = self._api_change_spec.module_deprecations
    if full_name in warnings:
      level, message = warnings[full_name]
      message = message.replace("<function name>", whole_name)
      self.add_log(level, node.lineno, node.col_offset,
                   "Using member %s in deprecated module %s. %s" % (whole_name,
                                                                    full_name,
                                                                    message))
      return True
    else:
      return False

  def _maybe_add_call_warning(self, node, full_name, name):
    # [FUNCTION DOC]
    # HELPER: Emits warnings for calls matching wildcard or argument-specific warning rules.
    # KNOWN ISSUE: the argument-presence test takes the result of the argument lookup, which is
    # a pair and therefore always true as a whole, so the fallback to the variadic flag never
    # applies; the message variant for calls with variadic arguments is only used when the
    # argument was actually found. Also the docstring refers to a nonexistent function name, and
    # a null short name with an attribute call would fail when building the wildcard key.
    # Inputs: node, call node; full_name, dotted callable name or nothing; name, short name or
    # nothing. First warns using the wildcard-dot-name key if the callee is an attribute, then
    # for each (keyword, position) rule applicable, in sorted order, warns when the argument is
    # present. Side effects: log entries, printed. Output: true if any warning was recorded.
    """Print a warning when specific functions are called with selected args.

    The function _print_warning_for_function matches the full name of the called
    function, e.g., tf.foo.bar(). This function matches the function name that
    is called, as long as the function is an attribute. For example,
    `tf.foo.bar()` and `foo.bar()` are matched, but not `bar()`.

    Args:
      node: ast.Call object
      full_name: The precomputed full name of the callable, if one exists, None
        otherwise.
      name: The precomputed name of the callable, if one exists, None otherwise.

    Returns:
      Whether an error was recorded.
    """
    # Only look for *.-warnings here, the other will be handled by the Attribute
    # visitor. Also, do not warn for bare functions, only if the call func is
    # an attribute.
    warned = False
    if isinstance(node.func, ast.Attribute):
      warned = self._maybe_add_warning(node, "*." + name)

    # All arg warnings are handled here, since only we have the args
    arg_warnings = self._get_applicable_dict("function_arg_warnings",
                                             full_name, name)

    variadic_args = uses_star_args_or_kwargs_in_call(node)

    for (kwarg, arg), (level, warning) in sorted(arg_warnings.items()):
      present, _ = get_arg_value(node, kwarg, arg) or variadic_args
      if present:
        warned = True
        warning_message = warning.replace("<function name>", full_name or name)
        template = "%s called with %s argument, requires manual check: %s"
        if variadic_args:
          template = ("%s called with *args or **kwargs that may include %s, "
                      "requires manual check: %s")
        self.add_log(level, node.lineno, node.col_offset,
                     template % (full_name or name, kwarg, warning_message))

    return warned

  def _maybe_rename(self, parent, node, full_name):
    # [FUNCTION DOC]
    # HELPER: Applies the symbol rename table to a name or attribute node.
    # Inputs: parent, the node containing it; node, the name/attribute node; full_name, its
    # dotted name. If a non-empty new name exists, logs an informational entry, builds a new
    # node chain at the same location with the same context, and replaces the node in the parent
    # in place. Output: true if renamed, else false.
    """Replace node (Attribute or Name) with a node representing full_name."""
    new_name = self._api_change_spec.symbol_renames.get(full_name, None)
    if new_name:
      self.add_log(INFO, node.lineno, node.col_offset,
                   "Renamed %r to %r" % (full_name, new_name))
      new_node = full_name_node(new_name, node.ctx)
      ast.copy_location(new_node, node)
      pasta.ast_utils.replace_child(parent, node, new_node)
      return True
    else:
      return False

  def _maybe_change_to_function_call(self, parent, node, full_name):
    # [FUNCTION DOC]
    # HELPER: Turns a property-like attribute access into a call when the specification lists
    # the name as changed to a function.
    # Inputs: parent, containing node; node, attribute node; full_name, its dotted name. If the
    # name is listed and the parent is not already a call, replaces the node by a call of it
    # with no arguments, copies the location, and logs an informational entry. Output: true if
    # changed, else false.
    """Wraps node (typically, an Attribute or Expr) in a Call."""
    if full_name in self._api_change_spec.change_to_function:
      if not isinstance(parent, ast.Call):
        # ast.Call's constructor is really picky about how many arguments it
        # wants, and also, it changed between Py2 and Py3.
        new_node = ast.Call(node, [], [])
        pasta.ast_utils.replace_child(parent, node, new_node)
        ast.copy_location(new_node, node)
        self.add_log(INFO, node.lineno, node.col_offset,
                     "Changed %r to a function call" % full_name)
        return True
    return False

  def _maybe_add_arg_names(self, node, full_name):
    # [FUNCTION DOC]
    # HELPER: Converts positional arguments into keyword arguments for functions whose argument
    # order changed.
    # KNOWN ISSUE: the lookup into the reorder list is not bounds checked, so a call with more
    # non-starred positional arguments than the list has entries raises an index error; also a
    # starred argument is skipped without advancing the position, so later positions can be
    # mapped to the wrong names.
    # Inputs: node, call node; full_name, dotted callable name or nothing. If the function is in
    # the reorder table, warns when starred positional arguments are used, then for each
    # positional argument looks up its new keyword (empty entries stay positional). If any
    # keyword was created, replaces the node's positional and keyword lists in place (new
    # keywords first) and logs. Output: true if modified, else false.
    """Make args into keyword args if function called full_name requires it."""
    function_reorders = self._api_change_spec.function_reorders

    if full_name in function_reorders:
      if uses_star_args_in_call(node):
        self.add_log(WARNING, node.lineno, node.col_offset,
                     "(Manual check required) upgrading %s may require "
                     "re-ordering the call arguments, but it was passed "
                     "variable-length positional *args. The upgrade "
                     "script cannot handle these automatically." % full_name)

      reordered = function_reorders[full_name]
      new_args = []
      new_keywords = []
      idx = 0
      for arg in node.args:
        if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
          continue  # Can't move Starred to keywords
        keyword_arg = reordered[idx]
        if keyword_arg:
          new_keywords.append(ast.keyword(arg=keyword_arg, value=arg))
        else:
          new_args.append(arg)
        idx += 1

      if new_keywords:
        self.add_log(INFO, node.lineno, node.col_offset,
                     "Added keywords to args of function %r" % full_name)
        node.args = new_args
        node.keywords = new_keywords + (node.keywords or [])
        return True
    return False

  def _maybe_modify_args(self, node, full_name, name):
    # [FUNCTION DOC]
    # HELPER: Renames or removes keyword arguments according to the keyword rename rules.
    # Inputs: node, call node; full_name, dotted callable name or nothing; name, short name.
    # Applicable rules map old keyword names to new names, where nothing/null means remove. If
    # the call uses double-star expansion, adds a manual-check warning. Matching keywords are
    # renamed in place or dropped, each with an informational log (printed); the node's keyword
    # list is replaced when anything changed. Output: true if modified, else false.
    """Rename keyword args if the function called full_name requires it."""
    renamed_keywords = self._get_applicable_dict("function_keyword_renames",
                                                 full_name, name)

    if not renamed_keywords:
      return False

    if uses_star_kwargs_in_call(node):
      self.add_log(WARNING, node.lineno, node.col_offset,
                   "(Manual check required) upgrading %s may require "
                   "renaming or removing call arguments, but it was passed "
                   "variable-length *args or **kwargs. The upgrade "
                   "script cannot handle these automatically." %
                   (full_name or name))
    modified = False
    new_keywords = []
    for keyword in node.keywords:
      argkey = keyword.arg
      if argkey in renamed_keywords:
        modified = True
        if renamed_keywords[argkey] is None:
          lineno = getattr(keyword, "lineno", node.lineno)
          col_offset = getattr(keyword, "col_offset", node.col_offset)
          self.add_log(INFO, lineno, col_offset,
                       "Removed argument %s for function %s" % (
                           argkey, full_name or name))
        else:
          keyword.arg = renamed_keywords[argkey]
          lineno = getattr(keyword, "lineno", node.lineno)
          col_offset = getattr(keyword, "col_offset", node.col_offset)
          self.add_log(INFO, lineno, col_offset,
                       "Renamed keyword argument for %s from %s to %s" % (
                           full_name, argkey, renamed_keywords[argkey]))
          new_keywords.append(keyword)
      else:
        new_keywords.append(keyword)

    if modified:
      node.keywords = new_keywords
    return modified

  def visit_Call(self, node):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor for call nodes. Input: node, the call node, which must be on top of the ancestor
    # stack (asserted). Computes the dotted and short callee names, then in order: emits call
    # warnings, converts arguments to keywords where required, renames or removes keywords,
    # warns if custom transformers exist and the call has variadic arguments, and runs each
    # applicable custom transformer with parent, node, names and a log list; transformer logs are
    # recorded and a returned different node replaces the node in the parent (and the stack top).
    # Finally visits children. Returns nothing; mutates the tree and the log.
    """Handle visiting a call node in the AST.

    Args:
      node: Current Node
    """
    assert self._stack[-1] is node

    # Get the name for this call, so we can index stuff with it.
    full_name = self._get_full_name(node.func)
    if full_name:
      name = full_name.split(".")[-1]
    elif isinstance(node.func, ast.Name):
      name = node.func.id
    elif isinstance(node.func, ast.Attribute):
      name = node.func.attr
    else:
      name = None

    # Call standard transformers for this node.
    # Make sure warnings come first, since args or names triggering warnings
    # may be removed by the other transformations.
    self._maybe_add_call_warning(node, full_name, name)
    # Make all args into kwargs
    self._maybe_add_arg_names(node, full_name)
    # Argument name changes or deletions
    self._maybe_modify_args(node, full_name, name)

    # Call transformers. These have the ability to modify the node, and if they
    # do, will return the new node they created (or the same node if they just
    # changed it). The are given the parent, but we will take care of
    # integrating their changes into the parent if they return a new node.
    #
    # These are matched on the old name, since renaming is performed by the
    # Attribute visitor, which happens later.
    transformers = self._get_applicable_entries("function_transformers",
                                                full_name, name)

    parent = self._stack[-2]

    if transformers:
      if uses_star_args_or_kwargs_in_call(node):
        self.add_log(WARNING, node.lineno, node.col_offset,
                     "(Manual check required) upgrading %s may require "
                     "modifying call arguments, but it was passed "
                     "variable-length *args or **kwargs. The upgrade "
                     "script cannot handle these automatically." %
                     (full_name or name))

    for transformer in transformers:
      logs = []
      new_node = transformer(parent, node, full_name, name, logs)
      self.add_logs(logs)
      if new_node and new_node is not node:
        pasta.ast_utils.replace_child(parent, node, new_node)
        node = new_node
        self._stack[-1] = node

    self.generic_visit(node)

  def visit_Attribute(self, node):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor for attribute nodes not handled as calls. Input: node, which must be on top of the
    # stack (asserted). If the dotted name can be built: emits a symbol warning, then tries to
    # rename the symbol or turn it into a function call (and stops descending if either
    # happened); otherwise finds the outermost enclosing attribute chain, dumps it to text, and
    # emits a module-deprecation warning. Then visits children. Returns nothing.
    """Handle bare Attributes i.e. [tf.foo, tf.bar]."""
    assert self._stack[-1] is node

    full_name = self._get_full_name(node)
    if full_name:
      parent = self._stack[-2]

      # Make sure the warning comes first, otherwise the name may have changed
      self._maybe_add_warning(node, full_name)

      # Once we did a modification, node is invalid and not worth inspecting
      # further. Also, we only perform modifications for simple nodes, so
      # There'd be no point in descending further.
      if self._maybe_rename(parent, node, full_name):
        return
      if self._maybe_change_to_function_call(parent, node, full_name):
        return

      # The isinstance check is enough -- a bare Attribute is never root.
      i = 2
      while isinstance(self._stack[-i], ast.Attribute):
        i += 1
      whole_name = pasta.dump(self._stack[-(i-1)])

      self._maybe_add_module_deprecation_warning(node, full_name, whole_name)

    self.generic_visit(node)

  def visit_Import(self, node):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor for plain import statements, applying the import rename table. Input: node, the
    # import node. For each imported module, tries the longest allowed dotted prefix first (up to
    # the specification's maximum submodule depth, default one) against the rename table, skipping
    # excluded names; on a match builds a renamed alias (a single-component import keeps its
    # original name as the alias), and inserts any follow-up statements configured for that import
    # right after it, copying location, indentation and ensuring a newline, with log entries.
    # If anything was renamed, replaces the import node and logs the change. Then visits children.
    # KNOWN ISSUE: after replacing the node the children of the old node are visited. Returns
    # nothing.
    """Handle visiting an import node in the AST.

    Args:
      node: Current Node
    """
    new_aliases = []
    import_updated = False
    import_renames = getattr(self._api_change_spec, "import_renames", {})
    max_submodule_depth = getattr(self._api_change_spec, "max_submodule_depth",
                                  1)
    inserts_after_imports = getattr(self._api_change_spec,
                                    "inserts_after_imports", {})

    # This loop processes imports in the format
    # import foo as f, bar as b
    for import_alias in node.names:
      all_import_components = import_alias.name.split(".")
      # Look for rename, starting with longest import levels.
      found_update = False
      for i in reversed(list(range(1, max_submodule_depth + 1))):
        import_component = all_import_components[0]
        for j in range(1, min(i, len(all_import_components))):
          import_component += "." + all_import_components[j]
        import_rename_spec = import_renames.get(import_component, None)

        if not import_rename_spec or excluded_from_module_rename(
            import_alias.name, import_rename_spec):
          continue

        new_name = (
            import_rename_spec.new_name +
            import_alias.name[len(import_component):])

        # If current import is
        #   import foo
        # then new import should preserve imported name:
        #   import new_foo as foo
        # This happens when module has just one component.
        new_asname = import_alias.asname
        if not new_asname and "." not in import_alias.name:
          new_asname = import_alias.name

        new_alias = ast.alias(name=new_name, asname=new_asname)
        new_aliases.append(new_alias)
        import_updated = True
        found_update = True

        # Insert any followup lines that should happen after this import.
        full_import = (import_alias.name, import_alias.asname)
        insert_offset = 1
        for line_to_insert in inserts_after_imports.get(full_import, []):
          assert self._stack[-1] is node
          parent = self._stack[-2]

          new_line_node = pasta.parse(line_to_insert)
          ast.copy_location(new_line_node, node)
          parent.body.insert(
              parent.body.index(node) + insert_offset, new_line_node)
          insert_offset += 1

          # Insert a newline after the import if necessary
          old_suffix = pasta.base.formatting.get(node, "suffix")
          if old_suffix is None:
            old_suffix = os.linesep
          if os.linesep not in old_suffix:
            pasta.base.formatting.set(node, "suffix", old_suffix + os.linesep)

          # Apply indentation to new node.
          pasta.base.formatting.set(new_line_node, "prefix",
                                    pasta.base.formatting.get(node, "prefix"))
          pasta.base.formatting.set(new_line_node, "suffix", os.linesep)
          self.add_log(
              INFO, node.lineno, node.col_offset,
              "Adding `%s` after import of %s" %
              (new_line_node, import_alias.name))
        # Find one match, break
        if found_update:
          break
      # No rename is found for all levels
      if not found_update:
        new_aliases.append(import_alias)  # no change needed

    # Replace the node if at least one import needs to be updated.
    if import_updated:
      assert self._stack[-1] is node
      parent = self._stack[-2]

      new_node = ast.Import(new_aliases)
      ast.copy_location(new_node, node)
      pasta.ast_utils.replace_child(parent, node, new_node)
      self.add_log(
          INFO, node.lineno, node.col_offset,
          "Changed import from %r to %r." %
          (pasta.dump(node), pasta.dump(new_node)))

    self.generic_visit(node)

  def visit_ImportFrom(self, node):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor for from-import statements, applying the import rename table. Input: node, the
    # statement node. Does nothing special for relative imports without a module name or when
    # the first module component has no rename rule. Otherwise splits the imported names into
    # those to move to the renamed module and those excluded by prefix; replaces the statement
    # with one using the renamed module for the first group, inserts an extra statement with the
    # original module for the excluded group before it (copying indentation), and logs the change.
    # Always ends by visiting children. Returns nothing; mutates the tree and the log.
    """Handle visiting an import-from node in the AST.

    Args:
      node: Current Node
    """
    if not node.module:
      self.generic_visit(node)
      return

    from_import = node.module

    # Look for rename based on first component of from-import.
    # i.e. based on foo in foo.bar.
    from_import_first_component = from_import.split(".")[0]
    import_renames = getattr(self._api_change_spec, "import_renames", {})
    import_rename_spec = import_renames.get(from_import_first_component, None)
    if not import_rename_spec:
      self.generic_visit(node)
      return

    # Split module aliases into the ones that require import update
    # and those that don't. For e.g. if we want to rename "a" to "b"
    # unless we import "a.c" in the following:
    # from a import c, d
    # we want to update import for "d" but not for "c".
    updated_aliases = []
    same_aliases = []
    for import_alias in node.names:
      full_module_name = "%s.%s" % (from_import, import_alias.name)
      if excluded_from_module_rename(full_module_name, import_rename_spec):
        same_aliases.append(import_alias)
      else:
        updated_aliases.append(import_alias)

    if not updated_aliases:
      self.generic_visit(node)
      return

    assert self._stack[-1] is node
    parent = self._stack[-2]

    # Replace first component of from-import with new name.
    new_from_import = (
        import_rename_spec.new_name +
        from_import[len(from_import_first_component):])
    updated_node = ast.ImportFrom(new_from_import, updated_aliases, node.level)
    ast.copy_location(updated_node, node)
    pasta.ast_utils.replace_child(parent, node, updated_node)

    # If some imports had to stay the same, add another import for them.
    additional_import_log = ""
    if same_aliases:
      same_node = ast.ImportFrom(from_import, same_aliases, node.level,
                                 col_offset=node.col_offset, lineno=node.lineno)
      ast.copy_location(same_node, node)
      parent.body.insert(parent.body.index(updated_node), same_node)
      # Apply indentation to new node.
      pasta.base.formatting.set(
          same_node, "prefix",
          pasta.base.formatting.get(updated_node, "prefix"))
      additional_import_log = " and %r" % pasta.dump(same_node)

    self.add_log(
        INFO, node.lineno, node.col_offset,
        "Changed import from %r to %r%s." %
        (pasta.dump(node),
         pasta.dump(updated_node),
         additional_import_log))

    self.generic_visit(node)


class AnalysisResult:
  # [CLASS DOC]
  # Marker/base class for analysis detections. Defines no members itself; implementations are
  # expected to provide a log level and a log message. Instances are produced by analysis
  # specifications and recorded by the analysis visitor.
  """This class represents an analysis result and how it should be logged.

  This class must provide the following fields:

  * `log_level`: The log level to which this detection should be logged
  * `log_message`: The message that should be logged for this detection

  For an example, see `VersionedTFImport`.
  """


class APIAnalysisSpec:
  # [CLASS DOC]
  # Base class for analysis specifications. Defines no members itself; subclasses provide a
  # mapping of symbol names to detections and a mapping of (module name, alias) pairs to
  # detections. KNOWN ISSUE: the docstring contains a stray dangling line about notifications
  # that does not describe any field.
  """This class defines how `AnalysisResult`s should be generated.

  It specifies how to map imports and symbols to `AnalysisResult`s.

  This class must provide the following fields:

  * `symbols_to_detect`: maps function names to `AnalysisResult`s
  * `imports_to_detect`: maps imports represented as (full module name, alias)
    tuples to `AnalysisResult`s
    notifications)

  For an example, see `TFAPIImportAnalysisSpec`.
  """


class PastaAnalyzeVisitor(_PastaEditVisitor):
  # [CLASS DOC]
  # Read-only visitor that scans a tree for specific symbols and imports listed in an analysis
  # specification and records detections plus log entries, without changing the tree. Extends
  # the edit visitor (using an empty change specification) and overrides the attribute and import
  # handlers. State: the analysis specification and a list of detections.
  """AST Visitor that looks for specific API usage without editing anything.

  This is used before any rewriting is done to detect if any symbols are used
  that require changing imports or disabling rewriting altogether.
  """

  def __init__(self, api_analysis_spec):
    # [FUNCTION DOC]
    # Constructor. Input: api_analysis_spec, the analysis specification. Initializes the parent
    # visitor with an empty change specification, stores the analysis spec and starts with an
    # empty results list. Returns nothing.
    super(PastaAnalyzeVisitor, self).__init__(NoUpdateSpec())
    self._api_analysis_spec = api_analysis_spec
    self._results = []   # Holds AnalysisResult objects

  @property
  def results(self):
    # [FUNCTION DOC]
    # Property returning the live list of detections recorded so far, in visit order.
    return self._results

  def add_result(self, analysis_result):
    # [FUNCTION DOC]
    # Appends a detection to the results list. Input: analysis_result, a detection object.
    # Returns nothing.
    self._results.append(analysis_result)

  def visit_Attribute(self, node):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor for attribute nodes in analysis mode. Input: node. If the dotted name matches an
    # entry in the specification's symbols-to-detect mapping, records the detection and adds a
    # log entry (printed) with its level and message. Then visits children. Does not modify the
    # tree and does not use the ancestor-stack assertions. Returns nothing.
    """Handle bare Attributes i.e. [tf.foo, tf.bar]."""
    full_name = self._get_full_name(node)
    if full_name:
      detection = self._api_analysis_spec.symbols_to_detect.get(full_name, None)
      if detection:
        self.add_result(detection)
        self.add_log(
            detection.log_level, node.lineno, node.col_offset,
            detection.log_message)

    self.generic_visit(node)

  def visit_Import(self, node):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor for plain imports in analysis mode. Input: node. For each imported name, looks up
    # the pair of full name and alias in the specification's imports-to-detect mapping; on a
    # match records the detection and logs it (printed). Then visits children. No tree changes.
    """Handle visiting an import node in the AST.

    Args:
      node: Current Node
    """
    for import_alias in node.names:
      # Detect based on full import name and alias)
      full_import = (import_alias.name, import_alias.asname)
      detection = (self._api_analysis_spec
                   .imports_to_detect.get(full_import, None))
      if detection:
        self.add_result(detection)
        self.add_log(
            detection.log_level, node.lineno, node.col_offset,
            detection.log_message)

    self.generic_visit(node)

  def visit_ImportFrom(self, node):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor for from-imports in analysis mode. Input: node. Ignores statements without a
    # module name. For each imported name, joins module and name with a dot, looks up that
    # string with the alias in the imports-to-detect mapping, and on a match records the
    # detection and logs it (printed). Then visits children. No tree changes.
    """Handle visiting an import-from node in the AST.

    Args:
      node: Current Node
    """
    if not node.module:
      self.generic_visit(node)
      return

    from_import = node.module

    for import_alias in node.names:
      # Detect based on full import name(to & as)
      full_module_name = "%s.%s" % (from_import, import_alias.name)
      full_import = (full_module_name, import_alias.asname)
      detection = (self._api_analysis_spec
                   .imports_to_detect.get(full_import, None))
      if detection:
        self.add_result(detection)
        self.add_log(
            detection.log_level, node.lineno, node.col_offset,
            detection.log_message)

    self.generic_visit(node)


class ASTCodeUpgrader:
  # [CLASS DOC]
  # Driver that upgrades Python source using a change specification: strings, single files,
  # opened file objects, and whole directory trees (in place or into a new output tree). Holds
  # the specification. Reports are strings; per-file errors are lists of formatted messages.
  """Handles upgrading a set of Python files using a given API change spec."""

  def __init__(self, api_change_spec):
    # [FUNCTION DOC]
    # Constructor. Input: api_change_spec, which must be an instance of the change specification
    # base class. Raises a type error otherwise. Stores it. Returns nothing.
    if not isinstance(api_change_spec, APIChangeSpec):
      raise TypeError("Must pass APIChangeSpec to ASTCodeUpgrader, got %s" %
                      type(api_change_spec))
    self._api_change_spec = api_change_spec

  def process_file(self,
                   in_filename,
                   out_filename,
                   no_change_to_outfile_on_error=False):
    # [FUNCTION DOC]
    # Upgrades one file on disk. Inputs: in_filename, path to read; out_filename, path to write
    # (may equal the input for in-place edits); no_change_to_outfile_on_error, flag, default false.
    # Writes the result to a temporary file first, then moves it to the output path. Output: the
    # triple (files processed count, report text, list of error messages) from the opened-file
    # step. KNOWN ISSUE: the flag name suggests keeping the output untouched on errors, but the
    # temporary file is discarded whenever the processed count is zero (i.e. parse failure), not
    # when warnings or errors were found; the move into place also happens otherwise. The
    # temporary file is created outside the destination directory, so the move may copy across
    # file systems. Raises file errors if files cannot be opened or moved.
    """Process the given python file for incompatible changes.

    Args:
      in_filename: filename to parse
      out_filename: output file to write to
      no_change_to_outfile_on_error: not modify the output file on errors
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """

    # Write to a temporary file, just in case we are doing an implace modify.
    # pylint: disable=g-backslash-continuation
    with open(in_filename, "r") as in_file, \
        tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
      ret = self.process_opened_file(in_filename, in_file, out_filename,
                                     temp_file)
    # pylint: enable=g-backslash-continuation

    if no_change_to_outfile_on_error and ret[0] == 0:
      os.remove(temp_file.name)
    else:
      shutil.move(temp_file.name, out_filename)
    return ret

  def format_log(self, log, in_filename):
    # [FUNCTION DOC]
    # Formats one log entry as text. Inputs: log, a (severity, line, column, message) tuple;
    # in_filename, optional file name (empty or nothing for none). Output: a line with line,
    # column, severity and message, prefixed by the file name and a colon when given.
    log_string = "%d:%d: %s: %s" % (log[1], log[2], log[0], log[3])
    if in_filename:
      return in_filename + ":" + log_string
    else:
      return log_string

  def update_string_pasta(self, text, in_filename):
    # [FUNCTION DOC]
    # Upgrades source text. Inputs: text, the source code; in_filename, used only to prefix
    # errors. Parses with pasta; on syntax, value or type errors returns a zero count, empty text,
    # a single log message with the traceback, and an empty error list. Otherwise runs the
    # specification's preprocessing hook, visits the tree with the edit visitor, calls the hook
    # that clears preprocessing, and returns the tuple (1, new source text, formatted log lines
    # without file names, formatted warning and error lines with the file name). Side effects: the
    # visitor prints log lines; the specification may be mutated by its hooks.
    """Updates a file using pasta."""
    try:
      t = pasta.parse(text)
    except (SyntaxError, ValueError, TypeError):
      log = ["ERROR: Failed to parse.\n" + traceback.format_exc()]
      return 0, "", log, []

    t, preprocess_logs, preprocess_errors = self._api_change_spec.preprocess(t)

    visitor = _PastaEditVisitor(self._api_change_spec)
    visitor.visit(t)

    self._api_change_spec.clear_preprocessing()

    logs = [self.format_log(log, None) for log in (preprocess_logs +
                                                   visitor.log)]
    errors = [self.format_log(error, in_filename)
              for error in (preprocess_errors +
                            visitor.warnings_and_errors)]
    return 1, pasta.dump(t), logs, errors

  def _format_log(self, log, in_filename, out_filename):
    # [FUNCTION DOC]
    # HELPER: Builds the per-file report block. Inputs: log, list of log lines; in_filename and
    # out_filename, names shown in the header. Output: a text with separator lines, a header
    # naming input and output files, the log lines one per line, and a closing separator.
    text = "-" * 80 + "\n"
    text += "Processing file %r\n outputting to %r\n" % (in_filename,
                                                         out_filename)
    text += "-" * 80 + "\n\n"
    text += "\n".join(log) + "\n"
    text += "-" * 80 + "\n\n"
    return text

  def process_opened_file(self, in_filename, in_file, out_filename, out_file):
    # [FUNCTION DOC]
    # Upgrades the contents of an already opened input. Inputs: in_filename and out_filename,
    # names used in reports; in_file, readable file-like object; out_file, writable file-like
    # object or nothing. Reads all lines, upgrades the text, and writes the new text to out_file
    # only when it exists and the text was successfully processed. Output: a triple of the
    # processed count (1 or 0), the formatted report text, and the list of error messages.
    """Process the given python file for incompatible changes.

    This function is split out to facilitate StringIO testing from
    tf_upgrade_test.py.

    Args:
      in_filename: filename to parse
      in_file: opened file (or StringIO)
      out_filename: output file to write to
      out_file: opened file (or StringIO)
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """
    lines = in_file.readlines()
    processed_file, new_file_content, log, process_errors = (
        self.update_string_pasta("".join(lines), in_filename))

    if out_file and processed_file:
      out_file.write(new_file_content)

    return (processed_file,
            self._format_log(log, in_filename, out_filename),
            process_errors)

  def process_tree(self, root_directory, output_root_directory,
                   copy_other_files):
    # [FUNCTION DOC]
    # Upgrades every Python file under a directory into a separate output tree. Inputs:
    # root_directory, the tree to read; output_root_directory, the new tree (if equal to the
    # input, delegates to the in-place variant); copy_other_files, whether to copy non-Python
    # files unchanged. Exits the whole process with status one (after printing a message) if the
    # output directory already exists or normalizes to the same path as the input. KNOWN ISSUE:
    # the docstring says the tree is processed in place, but this path writes to a new tree; the
    # overlap check compares only split parent/leaf pairs, so nested output directories are not
    # caught, and the final-name existing-directory check is run before the same-directory check
    # so that message is rarely reachable. Symbolic links to files in the tree are recreated as
    # links (retargeted if the target is also processed) rather than processed. Creates output
    # directories, writes files, copies other files. Output: triple of the number of processed
    # files, a report text, and a mapping from input path to its list of errors.
    """Processes upgrades on an entire tree of python files in place.

    Note that only Python files. If you have custom code in other languages,
    you will need to manually upgrade those.

    Args:
      root_directory: Directory to walk and process.
      output_root_directory: Directory to use as base.
      copy_other_files: Copy files that are not touched by this converter.

    Returns:
      A tuple of files processed, the report string for all files, and a dict
        mapping filenames to errors encountered in that file.
    """

    if output_root_directory == root_directory:
      return self.process_tree_inplace(root_directory)

    # make sure output directory doesn't exist
    if output_root_directory and os.path.exists(output_root_directory):
      print("Output directory %r must not already exist." %
            (output_root_directory))
      sys.exit(1)

    # make sure output directory does not overlap with root_directory
    norm_root = os.path.split(os.path.normpath(root_directory))
    norm_output = os.path.split(os.path.normpath(output_root_directory))
    if norm_root == norm_output:
      print("Output directory %r same as input directory %r" %
            (root_directory, output_root_directory))
      sys.exit(1)

    # Collect list of files to process (we do this to correctly handle if the
    # user puts the output directory in some sub directory of the input dir)
    files_to_process = []
    files_to_copy = []
    for dir_name, _, file_list in os.walk(root_directory):
      py_files = [f for f in file_list if f.endswith(".py")]
      copy_files = [f for f in file_list if not f.endswith(".py")]
      for filename in py_files:
        fullpath = os.path.join(dir_name, filename)
        fullpath_output = os.path.join(output_root_directory,
                                       os.path.relpath(fullpath,
                                                       root_directory))
        files_to_process.append((fullpath, fullpath_output))
      if copy_other_files:
        for filename in copy_files:
          fullpath = os.path.join(dir_name, filename)
          fullpath_output = os.path.join(output_root_directory,
                                         os.path.relpath(
                                             fullpath, root_directory))
          files_to_copy.append((fullpath, fullpath_output))

    file_count = 0
    tree_errors = {}
    report = ""
    report += ("=" * 80) + "\n"
    report += "Input tree: %r\n" % root_directory
    report += ("=" * 80) + "\n"

    for input_path, output_path in files_to_process:
      output_directory = os.path.dirname(output_path)
      if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

      if os.path.islink(input_path):
        link_target = os.readlink(input_path)
        link_target_output = os.path.join(
            output_root_directory, os.path.relpath(link_target, root_directory))
        if (link_target, link_target_output) in files_to_process:
          # Create a link to the new location of the target file
          os.symlink(link_target_output, output_path)
        else:
          report += "Copying symlink %s without modifying its target %s" % (
              input_path, link_target)
          os.symlink(link_target, output_path)
        continue

      file_count += 1
      _, l_report, l_errors = self.process_file(input_path, output_path)
      tree_errors[input_path] = l_errors
      report += l_report

    for input_path, output_path in files_to_copy:
      output_directory = os.path.dirname(output_path)
      if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
      shutil.copy(input_path, output_path)
    return file_count, report, tree_errors

  def process_tree_inplace(self, root_directory):
    # [FUNCTION DOC]
    # Upgrades every Python file under a directory, overwriting each file. Input:
    # root_directory. Symbolic links are skipped with a note in the report. Output: triple of
    # the number of files processed, the report text, and a mapping from file path to its list
    # of errors. Side effects: rewrites files on disk.
    """Process a directory of python files in place."""
    files_to_process = []
    for dir_name, _, file_list in os.walk(root_directory):
      py_files = [
          os.path.join(dir_name, f) for f in file_list if f.endswith(".py")
      ]
      files_to_process += py_files

    file_count = 0
    tree_errors = {}
    report = ""
    report += ("=" * 80) + "\n"
    report += "Input tree: %r\n" % root_directory
    report += ("=" * 80) + "\n"

    for path in files_to_process:
      if os.path.islink(path):
        report += "Skipping symlink %s.\n" % path
        continue
      file_count += 1
      _, l_report, l_errors = self.process_file(path, path)
      tree_errors[path] = l_errors
      report += l_report

    return file_count, report, tree_errors
