# [MODULE DOC]
# Static extractor for the second-generation API generator. Parses Python source files into
# syntax trees (without importing them) and records which symbols are exported to the public
# API through the export decorator (as decorator, as wrapping assignment, as wrapping call, or as
# constant export), plus top-level string literals marked as API docstrings. Results are
# accumulated into an exported-API collection and written to an output file. Exposes a parse
# error type for malformed exports, the Parser visitor class and a main function.
# Import-time side effects: defines three command-line flags (output file, decorator full path,
# API name prefix) and a compiled docstring-marker pattern.
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
# =============================================================================
"""Extracts API information for a set of Python sources."""

import ast
from collections.abc import Sequence
import re
from typing import Any, Optional, Union, cast

from absl import flags
from absl import logging

from tensorflow.python.tools.api.generator2.shared import exported_api

_OUTPUT = flags.DEFINE_string('output', '', 'File to output contents to.')
_DECORATOR = flags.DEFINE_string(
    'decorator',
    '',
    'Full path to Python decorator function used for exporting API.',
)
_API_NAME = flags.DEFINE_string(
    'api_name',
    '',
    'Prefix for all exported symbols and docstrings.',
)

_DOCSTRING_PATTERN: re.Pattern[str] = re.compile(
    r'\s*API\s+docstring:\s*([\w.]+)\s*'
)


class BadExportError(Exception):
  # [CLASS DOC]
  # Error raised when an export is used in an unsupported way (not at top level, wrong argument
  # shapes, non-literal arguments, unknown keywords, a dotted symbol name). Message includes the
  # file name, line number and a dump of the offending syntax node. No extra state.
  """Exception for bad exports."""


class Parser(ast.NodeVisitor):
  # [CLASS DOC]
  # Syntax-tree visitor that finds exported APIs in source files. State: the collection that
  # receives results, the package and symbol parts of the export decorator's full path, the API
  # name prefix, the file currently being processed and the set of local names under which the
  # decorator is reachable in that file (found from import statements). Used by creating one
  # instance and calling the file processing methods for each source. Top-level statements are
  # inspected specially; nested uses of the export are errors.
  """Parser for Python source files that extracts TF API exports."""

  _exports: exported_api.ExportedApi
  _decorator_package: str
  _decorator_symbol: str
  _api_name: str
  _current_file: Optional[str] = None
  _current_file_decorators: set[str]

  def __init__(
      self,
      exports: exported_api.ExportedApi,
      decorator: str,
      api_name: str,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: exports (collection that will receive exported symbols and docs),
    # decorator (full dotted path of the export decorator, must contain at least one dot, else
    # unpacking raises a value error) and api_name (prefix for all exported names). Splits the
    # decorator into package and symbol parts. Returns nothing.
    self._exports = exports
    self._decorator_package, self._decorator_symbol = decorator.rsplit('.', 1)
    self._api_name = api_name

  def process_file(self, filename: str) -> None:
    # [FUNCTION DOC]
    # Reads a source file as UTF-8 text and processes its contents. Input: file path. Any
    # exception while reading is logged and swallowed (the file is skipped). Exceptions from
    # processing, such as bad export errors, propagate. Returns nothing.
    """Finds exported APIs in filename."""
    try:
      with open(filename, mode='r', encoding='utf-8') as f:
        contents = f.read()
    except Exception as e:  # pylint: disable=broad-exception-caught
      # log and ignore exceptions from read
      logging.exception('Error reading %s: %s', filename, e)
    else:
      self.process(filename, contents)

  def process(self, filename: str, contents: str) -> None:
    # [FUNCTION DOC]
    # Parses source text and visits the tree to collect exports. Inputs: file name (for messages
    # and records) and the file contents. Resets the per-file decorator names and current file,
    # parses (a parse failure is logged and the file skipped), visits the tree, and in all cases
    # clears the per-file state afterwards. Bad export errors raised while visiting propagate.
    # Side effects: adds symbols and docs to the results collection; logging.
    """Finds exported APIs in contents."""
    self._current_file_decorators = set()
    self._current_file = filename
    try:
      parsed = ast.parse(contents, filename=filename)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # logging errors when parsing file
      logging.exception('Error parsing %s: %s', filename, e)
    else:
      self.visit(parsed)
    finally:
      self._current_file = None
      self._current_file_decorators = set()

  def visit_Module(self, node: ast.Module) -> None:  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor hook for the whole module. Passes each top-level statement to the statement
    # processor so top-level exports are recognized (nested ones are handled by other hooks).
    # Input: the module node. Returns nothing.
    for stmt in node.body:
      self._process_stmt(stmt)

  def _process_stmt(self, node: ast.stmt) -> None:
    # [FUNCTION DOC]
    # HELPER: dispatcher for top-level statements.
    # Sends class and function definitions, assignments and expression statements to their
    # dedicated processors; any other statement is visited generically. Input: statement node.
    # Returns nothing.
    """Process top-level statement for exported apis."""
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
      self._process_def(node)
    elif isinstance(node, ast.Assign):
      self._process_assign(node)
    elif isinstance(node, ast.Expr):
      self._process_expr(node)
    else:
      self.visit(node)

  def visit_Import(self, node: ast.Import) -> None:  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is generic. A plain import of the decorator's package registers
    # only the last segment of the package path plus the decorator name, which would not match a
    # use of the full dotted path in the source.
    # Visitor hook for import statements. When an imported name equals the decorator's package,
    # records the local name under which the decorator becomes reachable: the alias plus symbol
    # when aliased, otherwise the last package segment plus symbol. Then visits children.
    # Mutates the per-file decorator names. Input: import node. Returns nothing.
    """Identifies imports of decorator."""
    for name in node.names:
      if name.name == self._decorator_package:
        if name.asname:
          # import <package> as <name>
          self._current_file_decorators.add(
              name.asname + '.' + self._decorator_symbol
          )
        else:
          # import <package>
          _, module = self._decorator_package.rsplit('.', 1)
          self._current_file_decorators.add(
              module + '.' + self._decorator_symbol
          )
    self.generic_visit(node)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor hook for from-import statements. If the source module is the decorator's package
    # and the decorator symbol is imported, records its local name (alias or original). Otherwise,
    # if the source module is the parent of the decorator's package and the package's last segment
    # is imported, records the local name of that module joined with the symbol. Then visits
    # children. Mutates the per-file decorator names. Input: from-import node. Returns nothing.
    """Identifies imports of decorator."""
    if node.module == self._decorator_package:
      for name in node.names:
        if name.name == self._decorator_symbol:
          if name.asname:
            # from <package> import <symbol> as <name>
            self._current_file_decorators.add(name.asname)
          else:
            # from <package> import <symbol>
            self._current_file_decorators.add(name.name)
    else:
      parent, module = self._decorator_package.rsplit('.', 1)
      if node.module == parent:
        for name in node.names:
          if name.name == module:
            if name.asname:
              # from <parent> import <module> as <name>
              self._current_file_decorators.add(
                  name.asname + '.' + self._decorator_symbol
              )
            else:
              # from <parent> import <module>
              self._current_file_decorators.add(
                  name.name + '.' + self._decorator_symbol
              )
    self.generic_visit(node)

  def _process_def(self, node: Union[ast.ClassDef, ast.FunctionDef]) -> None:
    # [FUNCTION DOC]
    # KNOWN ISSUE: only plain function definitions are accepted by the dispatcher; async
    # function definitions fall to the generic visitor, so a decorator export on them is
    # reported as a not-top-level error.
    # Processes a top-level class or function definition. Each decorator that is an export call
    # registers the definition's name as an exported symbol; other decorators are visited. Then
    # visits base classes and keywords (classes) or arguments and return annotation (functions)
    # and every body statement, so nested misuse of the export is caught. Raises a bad export
    # error on invalid export calls. Returns nothing.
    """Process top-level [Class|Function]Def for potential symbol export."""
    # @tf_export(...)
    # [class|def] <id>:
    for decorator in node.decorator_list:
      if self._is_export_call(decorator):
        self._add_exported_symbol(cast(ast.Call, decorator), node.name)
      else:
        self.visit(decorator)

    if isinstance(node, ast.ClassDef):
      for base in node.bases:
        self.visit(base)
      for kw in node.keywords:
        self.visit(kw)
    elif isinstance(node, ast.FunctionDef):
      self.visit(node.args)
      if node.returns:
        self.visit(node.returns)

    for stmt in node.body:
      self.visit(stmt)

  def _process_assign(self, node: ast.Assign) -> None:
    # [FUNCTION DOC]
    # Processes a top-level assignment whose value is a call of the export call. Recognizes the
    # pattern of assigning the result of calling the export result to a single name, and registers
    # that name as exported. Raises a bad export error if there is more than one target or the
    # target is not a simple name (or attribute chain). Assignments of anything else are visited
    # generically. Input: assignment node. Returns nothing.
    """Process top-level assign for potential symbol export."""
    if isinstance(node.value, ast.Call) and self._is_export_call(
        node.value.func
    ):
      # id = tf_export(...)(...)
      if len(node.targets) != 1:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' assigned to a single value: {ast.dump(node)}'
        )
      symbol = self._name(node.targets[0])
      if not symbol:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' assigned to a single value: {ast.dump(node)}'
        )
      self._add_exported_symbol(node.value.func, symbol)
    else:
      self.visit(node)

  def _process_expr(self, node: ast.Expr) -> None:
    # [FUNCTION DOC]
    # Processes a top-level expression statement: calls go to the call processor, string or other
    # constants go to the constant processor (API docstring detection), everything else is
    # visited generically. Input: expression statement node. Returns nothing.
    """Process top-level expression for potential symbol export."""
    if isinstance(node.value, ast.Call):
      self._process_call(node.value)
    elif isinstance(node.value, ast.Constant):
      self._process_constant(node.value)
    else:
      self.visit(node)

  def _process_call(self, node: ast.Call) -> None:
    # [FUNCTION DOC]
    # Processes a top-level call. Two forms are exports: the export call applied to exactly one
    # positional argument (possibly wrapped in single-argument calls), whose resulting simple name
    # is exported; and the export call's constant-export method called with the current module
    # name placeholder and a literal constant name, which exports that literal. Raises a bad
    # export error when argument counts, keywords or the module-name argument are wrong, or no
    # simple name can be found. Other calls are visited generically. Returns nothing.
    """Process top-level call for potential symbol export."""
    func = node.func
    if self._is_export_call(func):
      func = cast(ast.Call, func)
      # tf_export(...)(id)
      if len(node.args) != 1 or node.keywords:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' called with a single value: {ast.dump(node)}'
        )
      symbol = self._name(self._unwrap_simple_call(node.args[0]))
      if not symbol:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' called with a single value: {ast.dump(node)}'
        )
      self._add_exported_symbol(func, symbol)
    elif (
        isinstance(func, ast.Attribute)
        and func.attr == 'export_constant'
        and self._is_export_call(func.value)
    ):
      # tf_export(...).export_constant(__name__, id)
      if (
          len(node.args) != 2
          or node.keywords
          or self._name(node.args[0]) != '__name__'
      ):
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export_constant must be'
            f' called with __name__, <id>: {ast.dump(node)}'
        )
      self._add_exported_symbol(func.value, self._literal_value(node.args[1]))
    else:
      self.visit(node)

  def _process_constant(self, node: ast.Constant) -> None:
    # [FUNCTION DOC]
    # KNOWN ISSUE: a non-string constant, or a string without any matching API docstring marker,
    # is only visited when the value is a string; non-string constants are silently ignored
    # (the visit sits inside the string check), which is harmless but inconsistent.
    # Processes a top-level constant. For a string, extracts the docstring text and target
    # modules; if any modules match the API name, adds an exported doc record (file, line,
    # modules, text) to the collection, otherwise visits the node generically. Returns nothing.
    """Process top-level constant for a potential API docstring export."""
    if isinstance(node.value, str):
      docstring, modules = self._extract_docstring(node.value)
      if modules:
        self._exports.add_doc(
            exported_api.ExportedDoc.create(
                file_name=self._current_file,
                line_no=node.lineno,
                modules=modules,
                docstring=docstring,
            )
        )
      else:
        self.visit(node)

  def _extract_docstring(self, value: str) -> tuple[str, Sequence[str]]:
    # [FUNCTION DOC]
    # Splits a string literal into docstring text and target modules. Each line matching the
    # API-docstring marker contributes its module name, kept only if it equals the API name or
    # starts with the API name plus a dot; all other lines (including marker lines for other
    # APIs are dropped, non-marker lines are kept) form the text. Input: string value. Returns a
    # pair of the stripped text and the list of matching module names. No side effects.
    """Extract docstring and list of modules that it should be applied to."""
    docstring = ''
    modules = []
    for line in value.splitlines():
      match = _DOCSTRING_PATTERN.match(line)
      if match:
        module = match.group(1).strip()
        # API docstring: <module>
        if module == self._api_name or module.startswith(self._api_name + '.'):
          modules.append(module)
      else:
        docstring += line + '\n'
    return (docstring.strip(), modules)

  def visit_Call(self, node: ast.Call) -> None:  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Visitor hook for calls reached outside the top-level patterns. Raises a bad export error if
    # the call is a use of the export (exports must be at top level); otherwise visits children.
    # Input: call node. Returns nothing.
    if self._is_export_call(node):
      raise BadExportError(
          f'{self._current_file}:{node.lineno} export must be'
          f' used at top level of file: {ast.dump(node)}'
      )
    self.generic_visit(node)

  def visit_Constant(self, node: ast.Constant) -> None:
    # [FUNCTION DOC]
    # Visitor hook for constants reached outside the top-level patterns. If a string constant
    # contains an API docstring marker for this API, raises a bad export error (such docstrings
    # must be at top level); otherwise visits children. Input: constant node. Returns nothing.
    if isinstance(node.value, str):
      _, modules = self._extract_docstring(node.value)
      if modules:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} API docstrings must be'
            f' at top level of file: {ast.dump(node)}'
        )
    self.generic_visit(node)

  def _is_export_call(self, node: ast.expr) -> bool:  # TypeGuard[ast.Call]
    # [FUNCTION DOC]
    # HELPER: predicate used by the statement processors.
    # Returns true if the node is a call whose callee's dotted name is one of the local names
    # under which the export decorator was imported in the current file. Input: expression node.
    # No side effects.
    return (
        isinstance(node, ast.Call)
        and self._name(node.func) in self._current_file_decorators
    )

  def _name(self, node: ast.expr) -> Optional[str]:
    # [FUNCTION DOC]
    # HELPER: name resolution for the statement processors.
    # Returns the identifier for a simple name node, or the dotted path for a chain of attribute
    # accesses on names; returns nothing (null) for any other kind of expression or a chain that
    # does not start with a name. Input: expression node. Pure function.
    if isinstance(node, ast.Name):
      return node.id
    if isinstance(node, ast.Attribute):
      parent = self._name(node.value)
      if parent:
        return f'{parent}.{node.attr}'

  def _unwrap_simple_call(self, node: ast.expr) -> ast.expr:
    # [FUNCTION DOC]
    # HELPER: argument normalizer for the call processor.
    # Repeatedly unwraps calls that have exactly one positional argument and no keywords,
    # returning the innermost argument; any other expression is returned unchanged. Input:
    # expression node. Pure function.
    """Unwraps a function call that takes a single unnamed parameter."""
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
      return self._unwrap_simple_call(node.args[0])
    return node

  def _literal_value(self, node: ast.expr) -> Any:
    # [FUNCTION DOC]
    # HELPER: literal evaluation for export arguments.
    # Evaluates the expression as a literal (strings, numbers, tuples, lists and so on) without
    # running code. Input: expression node. Returns the value. Raises a bad export error (chained
    # to the original failure) if the expression is not a literal.
    try:
      return ast.literal_eval(node)
    except Exception as e:
      raise BadExportError(
          f'{self._current_file}:{node.lineno} all arguments to'
          f' export must be literal values: {ast.dump(node)}'
      ) from e

  def _add_exported_symbol(self, node: ast.Call, symbol_name: str) -> None:
    # [FUNCTION DOC]
    # Records one exported symbol. Inputs: the export call node and the symbol's local name.
    # Positional arguments (literals) become the v2 API names, each prefixed with the API name;
    # a keyword for v1 supplies a separate list of v1 names (otherwise v1 equals v2); a keyword
    # for allowing multiple exports is accepted and ignored. Raises a bad export error if the
    # symbol name contains a dot (symbol not defined in this file), if an argument is not a
    # literal, or for any other keyword. Adds an exported symbol record (file, line, name, v2
    # names, v1 names) to the results collection. Returns nothing.
    """Adds an exported symbol represented by the given call."""
    if symbol_name.find('.') != -1:
      raise BadExportError(
          f'{self._current_file}:{node.lineno} export called with symbol'
          f' {symbol_name} not defined in current file: {ast.dump(node)}'
      )
    v2_apis = tuple(
        f'{self._api_name}.{self._literal_value(arg)}' for arg in node.args
    )
    v1_apis = v2_apis
    for kw in node.keywords:
      if kw.arg == 'v1':
        v1_apis = tuple(
            f'{self._api_name}.{v}' for v in self._literal_value(kw.value)
        )
      elif kw.arg == 'allow_multiple_exports':
        # no-op kept for backward compatibility of `tf-keras` with TF 2.13
        pass
      else:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export called'
            f' with unknown argument {kw.arg}: {ast.dump(node)}'
        )
    self._exports.add_symbol(
        exported_api.ExportedSymbol.create(
            file_name=self._current_file,
            line_no=node.lineno,
            symbol_name=symbol_name,
            v2_apis=v2_apis,
            v1_apis=v1_apis,
        )
    )


def main(argv: Sequence[str]) -> None:
  # [FUNCTION DOC]
  # Entry point run by the application runner. Input: argument list whose first item is the
  # program name and the rest are source file paths. Creates an empty results collection and a
  # parser configured from the decorator and API-name flags, processes every source file, then
  # writes the collection to the output flag's file path. Side effects: reads files, writes the
  # output file, logs read/parse failures. Bad export errors propagate. Returns nothing.
  exporter = exported_api.ExportedApi()
  p = Parser(exporter, _DECORATOR.value, _API_NAME.value)
  for arg in argv[1:]:
    p.process_file(arg)

  exporter.write(_OUTPUT.value)
