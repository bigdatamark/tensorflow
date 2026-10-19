# [MODULE DOC]
# Human-readable dump of syntax trees, a more readable alternative to the standard tree dump.
# Exposes the PrettyPrinter visitor class and the fmt convenience function, which renders a node
# (or list/tuple of nodes) as an indented multi-line text showing node type names, field names
# and values, optionally with terminal colors. Used for debugging and for failure messages in
# autograph tests and source-map code. Depends on the gast and termcolor libraries. No
# import-time side effects.
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
"""Print an AST tree in a form more readable than ast.dump."""

import gast
import termcolor


class PrettyPrinter(gast.NodeVisitor):
  # [CLASS DOC]
  # Visitor that accumulates a textual tree dump of syntax nodes. State: current indentation
  # level, the accumulated result text, a flag for colored output and a flag for hiding
  # annotation fields (those whose names start with double underscore). Created by fmt; each
  # visited node appends lines to the result, which the caller reads afterwards.
  """Print AST nodes."""

  def __init__(self, color, noanno):
    # [FUNCTION DOC]
    # Takes color (true to emit terminal color codes) and noanno (true to skip fields whose names
    # start with double underscore). Starts with indentation level zero and an empty result text.
    self.indent_lvl = 0
    self.result = ''
    self.color = color
    self.noanno = noanno

  def _color(self, string, color, attrs=None):
    # [FUNCTION DOC]
    # HELPER: styling primitive for the printer.
    # Returns string wrapped in terminal color codes for the given color name and optional list of
    # attribute names when coloring is enabled; otherwise returns string unchanged.
    if self.color:
      return termcolor.colored(string, color, attrs=attrs)
    return string

  def _type(self, node):
    # [FUNCTION DOC]
    # HELPER: styling for the printer. Returns the node's class name, in bold when coloring is on.
    return self._color(node.__class__.__name__, None, ['bold'])

  def _field(self, name):
    # [FUNCTION DOC]
    # HELPER: styling for the printer. Returns a field name, blue when coloring is on.
    return self._color(name, 'blue')

  def _value(self, name):
    # [FUNCTION DOC]
    # HELPER: styling for the printer. Returns a field value (text), magenta when coloring is on.
    # KNOWN ISSUE: the parameter is called name, but it receives values, which may be non-text
    # objects; with coloring on these are passed to the color library unconverted.
    return self._color(name, 'magenta')

  def _warning(self, name):
    # [FUNCTION DOC]
    # HELPER: styling for the printer. Returns warning text, red when coloring is on.
    return self._color(name, 'red')

  def _indent(self):
    # [FUNCTION DOC]
    # HELPER: layout for the printer. Returns the indentation prefix: a vertical bar and space
    # repeated once per current indentation level, dimmed when coloring is on.
    return self._color('| ' * self.indent_lvl, None, ['dark'])

  def _print(self, s):
    # [FUNCTION DOC]
    # HELPER: output primitive for the printer. Appends text s and a newline to the accumulated
    # result (mutates internal state; returns nothing).
    self.result += s
    self.result += '\n'

  def generic_visit(self, node, name=None):
    # [FUNCTION DOC]
    # Appends a dump of node and its children to the result. Inputs: node (a syntax node, or a
    # plain string found in a list field such as global names) and name (optional field name under
    # which the node is shown).
    # Strings are printed quoted, with the field name if given. For nodes, prints the field name
    # (if any), the node type and a colon (or empty parentheses if the type has no fields), then
    # raises the indentation level and prints each field: fields starting with double underscore
    # are skipped when annotation hiding is on; unset fields are shown as unset in a warning
    # style; lists and tuples are printed in brackets or parentheses with each element dumped
    # recursively (null elements shown as None; empty ones shown empty); nested nodes are dumped
    # recursively; bytes and text values are shown quoted with a b or u prefix; other values as is.
    # Restores the indentation level and returns nothing.
    # Elements of lists and tuples that are neither nodes, strings nor null (e.g. numbers) would
    # fail on the missing field-list attribute.
    # In very rare instances, a list can contain something other than a Node.
    # e.g. Global contains a list of strings.
    if isinstance(node, str):
      if name:
        self._print('%s%s="%s"' % (self._indent(), name, node))
      else:
        self._print('%s"%s"' % (self._indent(), node))
      return

    if node._fields:
      cont = ':'
    else:
      cont = '()'

    if name:
      self._print('%s%s=%s%s' % (self._indent(), self._field(name),
                                 self._type(node), cont))
    else:
      self._print('%s%s%s' % (self._indent(), self._type(node), cont))

    self.indent_lvl += 1
    for f in node._fields:
      if self.noanno and f.startswith('__'):
        continue
      if not hasattr(node, f):
        self._print('%s%s' % (self._indent(), self._warning('%s=<unset>' % f)))
        continue
      v = getattr(node, f)
      if isinstance(v, list):
        if v:
          self._print('%s%s=[' % (self._indent(), self._field(f)))
          self.indent_lvl += 1
          for n in v:
            if n is not None:
              self.generic_visit(n)
            else:
              self._print('%sNone' % (self._indent()))
          self.indent_lvl -= 1
          self._print('%s]' % (self._indent()))
        else:
          self._print('%s%s=[]' % (self._indent(), self._field(f)))
      elif isinstance(v, tuple):
        if v:
          self._print('%s%s=(' % (self._indent(), self._field(f)))
          self.indent_lvl += 1
          for n in v:
            if n is not None:
              self.generic_visit(n)
            else:
              self._print('%sNone' % (self._indent()))
          self.indent_lvl -= 1
          self._print('%s)' % (self._indent()))
        else:
          self._print('%s%s=()' % (self._indent(), self._field(f)))
      elif isinstance(v, gast.AST):
        self.generic_visit(v, f)
      elif isinstance(v, bytes):
        self._print('%s%s=%s' % (self._indent(), self._field(f),
                                 self._value('b"%s"' % v)))
      elif isinstance(v, str):
        self._print('%s%s=%s' % (self._indent(), self._field(f),
                                 self._value('u"%s"' % v)))
      else:
        self._print('%s%s=%s' % (self._indent(), self._field(f),
                                 self._value(v)))
    self.indent_lvl -= 1


def fmt(node, color=True, noanno=False):
  # [FUNCTION DOC]
  # Public entry point: renders a syntax node, or a list/tuple of nodes, as an indented text dump.
  # Inputs: node, color (default true; emit terminal color codes) and noanno (default false;
  # hide annotation fields starting with double underscore). Creates a printer, visits the node
  # (each element in turn for a sequence) and returns the accumulated text. No other side effects.
  printer = PrettyPrinter(color, noanno)
  if isinstance(node, (list, tuple)):
    for n in node:
      printer.visit(n)
  else:
    printer.visit(node)
  return printer.result
