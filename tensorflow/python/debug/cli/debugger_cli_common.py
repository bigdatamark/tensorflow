# [MODULE DOC]
# Building blocks shared by the TensorFlow debugger (tfdbg) command-line interfaces.
# Exposes: RichLine and RichTextLines (text with per-segment font attributes and per-row
# annotations that a terminal UI can render), helpers to search/wrap/convert such text,
# CommandHandlerRegistry (maps command prefixes and aliases to handler callables and
# dispatches commands, with built-in help and version commands), TabCompletionRegistry
# (context-word based tab completion candidates), CommandHistory (bounded, file-backed
# history of commands), MenuItem/Menu (text menus), CommandLineExit (exception used by
# handlers to request exit) and a few module constants (help indent, annotation keys, main
# menu key). Defining the module has no import-time side effects beyond defining these names.
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
"""Building Blocks of TensorFlow Debugger Command-Line Interface."""
import copy
import os
import re
import traceback

import numpy as np

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile

HELP_INDENT = "  "

EXPLICIT_USER_EXIT = "explicit_user_exit"
REGEX_MATCH_LINES_KEY = "regex_match_lines"
INIT_SCROLL_POS_KEY = "init_scroll_pos"

MAIN_MENU_KEY = "mm:"


class CommandLineExit(Exception):
  # [CLASS DOC]
  # Exception that a command handler raises to ask the command-line interface to exit.
  # It optionally carries an exit token of arbitrary type, readable through a property.
  # The registry's dispatch method re-raises it unchanged so the UI loop can catch it.

  def __init__(self, exit_token=None):
    # [FUNCTION DOC]
    # Constructs the exception. exit_token is an optional value of any type (default null)
    # stored for later retrieval. Calls the base exception constructor without a message.
    # Returns nothing.
    Exception.__init__(self)
    self._exit_token = exit_token

  @property
  def exit_token(self):
    # [FUNCTION DOC]
    # Property returning the exit token given at construction (null if none was given).
    return self._exit_token


class RichLine:
  # [CLASS DOC]
  # A single line of text plus a list of font-attribute segments (start, end, attribute).
  # Holds the public attributes text and font_attr_segs. Supports concatenation with plain
  # strings or other RichLine objects (offsetting segments) and length as text length.
  """Rich single-line text.

  Attributes:
    text: A plain string, the raw text represented by this object.  Should not
      contain newlines.
    font_attr_segs: A list of (start, end, font attribute) triples, representing
      richness information applied to substrings of text.
  """

  def __init__(self, text="", font_attr=None):
    # [FUNCTION DOC]
    # Builds a RichLine. text is the raw string (default empty, should hold no newlines).
    # font_attr is an optional single font attribute; if given (truthy), one segment covering
    # the whole text is created, otherwise the segment list is empty. Returns nothing.
    """Construct a RichLine with no rich attributes or a single attribute.

    Args:
      text: Raw text string
      font_attr: If specified, a single font attribute to be applied to the
        entire text.  Extending this object via concatenation allows creation
        of text with varying attributes.
    """
    # TODO(ebreck) Make .text and .font_attr protected members when we no
    # longer need public access.
    self.text = text
    if font_attr:
      self.font_attr_segs = [(0, len(text), font_attr)]
    else:
      self.font_attr_segs = []

  def __add__(self, other):
    # [FUNCTION DOC]
    # Concatenation operator. other may be a plain string (appended with no attributes) or a
    # RichLine (appended with its segments shifted by this line's text length). Returns a new
    # RichLine and leaves this one unchanged. Raises a type error for any other type.
    """Concatenate two chunks of maybe rich text to make a longer rich line.

    Does not modify self.

    Args:
      other: Another piece of text to concatenate with this one.
        If it is a plain str, it will be appended to this string with no
        attributes.  If it is a RichLine, it will be appended to this string
        with its attributes preserved.

    Returns:
      A new RichLine comprising both chunks of text, with appropriate
        attributes applied to the corresponding substrings.
    """
    ret = RichLine()
    if isinstance(other, str):
      ret.text = self.text + other
      ret.font_attr_segs = self.font_attr_segs[:]
      return ret
    elif isinstance(other, RichLine):
      ret.text = self.text + other.text
      ret.font_attr_segs = self.font_attr_segs[:]
      old_len = len(self.text)
      for start, end, font_attr in other.font_attr_segs:
        ret.font_attr_segs.append((old_len + start, old_len + end, font_attr))
      return ret
    else:
      raise TypeError("%r cannot be concatenated with a RichLine" % other)

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of characters in the raw text.
    return len(self.text)


def rich_text_lines_from_rich_line_list(rich_text_list, annotations=None):
  # [FUNCTION DOC]
  # Converts a list whose items are RichLine objects or plain strings into one RichTextLines.
  # Each item contributes one line; for RichLine items the text is used and any non-empty
  # segment list is recorded under that line's index. annotations is an optional mapping
  # passed through to the result. Returns the new RichTextLines. Raises nothing itself.
  """Convert a list of RichLine objects or strings to a RichTextLines object.

  Args:
    rich_text_list: a list of RichLine objects or strings
    annotations: annotations for the resultant RichTextLines object.

  Returns:
    A corresponding RichTextLines object.
  """
  lines = []
  font_attr_segs = {}
  for i, rl in enumerate(rich_text_list):
    if isinstance(rl, RichLine):
      lines.append(rl.text)
      if rl.font_attr_segs:
        font_attr_segs[i] = rl.font_attr_segs
    else:
      lines.append(rl)
  return RichTextLines(lines, font_attr_segs, annotations=annotations)


def get_tensorflow_version_lines(include_dependency_versions=False):
  # [FUNCTION DOC]
  # Builds a RichTextLines with the TensorFlow version (read from the pywrap session module),
  # followed by an empty line. If include_dependency_versions is true (default false) it adds
  # a header and the numpy version, then another empty line. Returns the RichTextLines; no
  # side effects.
  """Generate RichTextLines with TensorFlow version info.

  Args:
    include_dependency_versions: Include the version of TensorFlow's key
      dependencies, such as numpy.

  Returns:
    A formatted, multi-line `RichTextLines` object.
  """
  lines = ["TensorFlow version: %s" % pywrap_tf_session.__version__]
  lines.append("")
  if include_dependency_versions:
    lines.append("Dependency version(s):")
    lines.append("  numpy: %s" % np.__version__)
    lines.append("")
  return RichTextLines(lines)


class RichTextLines:
  # [CLASS DOC]
  # Multi-line text output of a debugger CLI command. Holds a list of lines, a mapping from
  # row index to font-attribute segments, and a mapping from row index (or other keys) to
  # annotations. Provides slicing, extending, appending, prepending and writing to a file.
  # Created by command handlers and rendered by UIs such as the curses interface.
  """Rich multi-line text.

  Line-by-line text output, with font attributes (e.g., color) and annotations
  (e.g., indices in a multi-dimensional tensor). Used as the text output of CLI
  commands. Can be rendered on terminal environments such as curses.

  This is not to be confused with Rich Text Format (RTF). This class is for text
  lines only.
  """

  def __init__(self, lines, font_attr_segs=None, annotations=None):
    # [FUNCTION DOC]
    # Constructs the object. lines is a list of strings or a single string (wrapped into a
    # one-element list). font_attr_segs and annotations are optional mappings from row index to
    # segments / annotation objects; null or empty values are replaced by new empty mappings.
    # The given list and mappings are stored directly, not copied. Raises a value error if
    # lines is neither a list nor a string. Returns nothing.
    """Constructor of RichTextLines.

    Args:
      lines: A list of str or a single str, representing text output to
        screen. The latter case is for convenience when the text output is
        single-line.
      font_attr_segs: A map from 0-based row index to a list of 3-tuples.
        It lists segments in each row that have special font attributes, such
        as colors, that are not the default attribute. For example:
        {1: [(0, 3, "red"), (4, 7, "green")], 2: [(10, 20, "yellow")]}

        In each tuple, the 1st element is the start index of the segment. The
        2nd element is the end index, in an "open interval" fashion. The 3rd
        element is an object or a list of objects that represents the font
        attribute. Colors are represented as strings as in the examples above.
      annotations: A map from 0-based row index to any object for annotating
        the row. A typical use example is annotating rows of the output as
        indices in a multi-dimensional tensor. For example, consider the
        following text representation of a 3x2x2 tensor:
          [[[0, 0], [0, 0]],
           [[0, 0], [0, 0]],
           [[0, 0], [0, 0]]]
        The annotation can indicate the indices of the first element shown in
        each row, i.e.,
          {0: [0, 0, 0], 1: [1, 0, 0], 2: [2, 0, 0]}
        This information can make display of tensors on screen clearer and can
        help the user navigate (scroll) to the desired location in a large
        tensor.

    Raises:
      ValueError: If lines is of invalid type.
    """
    if isinstance(lines, list):
      self._lines = lines
    elif isinstance(lines, str):
      self._lines = [lines]
    else:
      raise ValueError("Unexpected type in lines: %s" % type(lines))

    self._font_attr_segs = font_attr_segs
    if not self._font_attr_segs:
      self._font_attr_segs = {}
      # TODO(cais): Refactor to collections.defaultdict(list) to simplify code.

    self._annotations = annotations
    if not self._annotations:
      self._annotations = {}
      # TODO(cais): Refactor to collections.defaultdict(list) to simplify code.

  @property
  def lines(self):
    # [FUNCTION DOC]
    # Property returning the underlying (mutable) list of text lines.
    return self._lines

  @property
  def font_attr_segs(self):
    # [FUNCTION DOC]
    # Property returning the underlying mapping from row index to font attribute segments.
    return self._font_attr_segs

  @property
  def annotations(self):
    # [FUNCTION DOC]
    # Property returning the underlying mapping of annotations (row indices and other keys).
    return self._annotations

  def num_lines(self):
    # [FUNCTION DOC]
    # Returns the number of text lines held.
    return len(self._lines)

  def slice(self, begin, end):
    # [FUNCTION DOC]
    # Returns a new RichTextLines holding lines from begin (inclusive) to end (exclusive),
    # with segment and annotation row indices re-based to start at zero. Non-integer
    # annotation keys are copied over unchanged. The original is not modified. Raises a
    # value error if begin or end is negative.
    """Slice a RichTextLines object.

    The object itself is not changed. A sliced instance is returned.

    Args:
      begin: (int) Beginning line index (inclusive). Must be >= 0.
      end: (int) Ending line index (exclusive). Must be >= 0.

    Returns:
      (RichTextLines) Sliced output instance of RichTextLines.

    Raises:
      ValueError: If begin or end is negative.
    """

    if begin < 0 or end < 0:
      raise ValueError("Encountered negative index.")

    # Copy lines.
    lines = self.lines[begin:end]

    # Slice font attribute segments.
    font_attr_segs = {}
    for key in self.font_attr_segs:
      if key >= begin and key < end:
        font_attr_segs[key - begin] = self.font_attr_segs[key]

    # Slice annotations.
    annotations = {}
    for key in self.annotations:
      if not isinstance(key, int):
        # Annotations can contain keys that are not line numbers.
        annotations[key] = self.annotations[key]
      elif key >= begin and key < end:
        annotations[key - begin] = self.annotations[key]

    return RichTextLines(
        lines, font_attr_segs=font_attr_segs, annotations=annotations)

  def extend(self, other):
    # [FUNCTION DOC]
    # Appends another RichTextLines to this one in place: its lines are added at the end, and
    # its segment and integer-keyed annotation indices are shifted by this object's original
    # line count. Other annotation keys are copied, overriding existing ones. Returns nothing.
    """Extend this instance of RichTextLines with another instance.

    The extension takes effect on the text lines, the font attribute segments,
    as well as the annotations. The line indices in the font attribute
    segments and the annotations are adjusted to account for the existing
    lines. If there are duplicate, non-line-index fields in the annotations,
    the value from the input argument "other" will override that in this
    instance.

    Args:
      other: (RichTextLines) The other RichTextLines instance to be appended at
        the end of this instance.
    """

    orig_num_lines = self.num_lines()  # Record original number of lines.

    # Merge the lines.
    self._lines.extend(other.lines)

    # Merge the font_attr_segs.
    for line_index in other.font_attr_segs:
      self._font_attr_segs[orig_num_lines + line_index] = (
          other.font_attr_segs[line_index])

    # Merge the annotations.
    for key in other.annotations:
      if isinstance(key, int):
        self._annotations[orig_num_lines + key] = (other.annotations[key])
      else:
        self._annotations[key] = other.annotations[key]

  def _extend_before(self, other):
    # [FUNCTION DOC]
    # HELPER: Implements prepending for RichTextLines (used by the prepend method).
    # KNOWN ISSUE: for non-integer annotation keys of this object, the value is looked up in
    # the other object's annotations instead of this one's, which can fail with a missing-key
    # error or copy the wrong value.
    # Puts the other object's lines in front of this one's, in place. This object's segment
    # and integer annotation indices are shifted by the other's line count, then the other's
    # segments and annotations are merged in. Returns nothing.
    """Add another RichTextLines object to the front.

    Args:
      other: (RichTextLines) The other object to add to the front to this
        object.
    """

    other_num_lines = other.num_lines()  # Record original number of lines.

    # Merge the lines.
    self._lines = other.lines + self._lines

    # Merge the font_attr_segs.
    new_font_attr_segs = {}
    for line_index in self.font_attr_segs:
      new_font_attr_segs[other_num_lines + line_index] = (
          self.font_attr_segs[line_index])
    new_font_attr_segs.update(other.font_attr_segs)
    self._font_attr_segs = new_font_attr_segs

    # Merge the annotations.
    new_annotations = {}
    for key in self._annotations:
      if isinstance(key, int):
        new_annotations[other_num_lines + key] = (self.annotations[key])
      else:
        new_annotations[key] = other.annotations[key]

    new_annotations.update(other.annotations)
    self._annotations = new_annotations

  def append(self, line, font_attr_segs=None):
    # [FUNCTION DOC]
    # Adds one line (a string) at the end in place. If font_attr_segs (optional list of
    # segments) is non-empty it is recorded for the new last row. Returns nothing.
    """Append a single line of text.

    Args:
      line: (str) The text to be added to the end.
      font_attr_segs: (list of tuples) Font attribute segments of the appended
        line.
    """

    self._lines.append(line)
    if font_attr_segs:
      self._font_attr_segs[len(self._lines) - 1] = font_attr_segs

  def append_rich_line(self, rich_line):
    # [FUNCTION DOC]
    # Appends a RichLine as a new last line, carrying over its font attribute segments.
    # Returns nothing.
    self.append(rich_line.text, rich_line.font_attr_segs)

  def prepend(self, line, font_attr_segs=None):
    # [FUNCTION DOC]
    # Adds one line (a string) at the front in place, with optional font attribute segments for
    # it. Builds a temporary one-line object and merges it using the internal prepend helper,
    # so all existing indices shift by one. Returns nothing.
    """Prepend (i.e., add to the front) a single line of text.

    Args:
      line: (str) The text to be added to the front.
      font_attr_segs: (list of tuples) Font attribute segments of the appended
        line.
    """

    other = RichTextLines(line)
    if font_attr_segs:
      other.font_attr_segs[0] = font_attr_segs
    self._extend_before(other)

  def write_to_file(self, file_path):
    # [FUNCTION DOC]
    # Writes all lines to the file at file_path (a string path, opened through TensorFlow's
    # file API in write mode, overwriting), one per line with a newline appended. Font
    # attributes and annotations are not written. Side effect: creates or overwrites the file.
    # Returns nothing; file errors propagate.
    """Write the object itself to file, in a plain format.

    The font_attr_segs and annotations are ignored.

    Args:
      file_path: (str) path of the file to write to.
    """

    with gfile.Open(file_path, "w") as f:
      for line in self._lines:
        f.write(line + "\n")

  # TODO(cais): Add a method to allow appending to a line in RichTextLines with
  # both text and font_attr_segs.


def regex_find(orig_screen_output, regex, font_attr):
  # [FUNCTION DOC]
  # Searches every line of a RichTextLines for a regular expression and highlights matches.
  # Inputs: orig_screen_output (RichTextLines), regex (pattern string), font_attr (attribute
  # applied to each match). Returns a new RichTextLines sharing the same lines list and the
  # same annotations mapping but with a deep copy of the segments, containing extra segments
  # for matches (kept sorted by start when merged into existing ones). The annotation under the
  # regex-match-lines key is set to the list of matching row indices.
  # KNOWN ISSUE: the annotations mapping is not copied, so setting the match-lines key also
  # mutates the original object's annotations (despite the "modified copy" wording).
  # Raises a value error if the pattern is not a valid regular expression.
  """Perform regex match in rich text lines.

  Produces a new RichTextLines object with font_attr_segs containing highlighted
  regex matches.

  Example use cases include:
  1) search for specific items in a large list of items, and
  2) search for specific numerical values in a large tensor.

  Args:
    orig_screen_output: The original RichTextLines, in which the regex find
      is to be performed.
    regex: The regex used for matching.
    font_attr: Font attribute used for highlighting the found result.

  Returns:
    A modified copy of orig_screen_output.

  Raises:
    ValueError: If input str regex is not a valid regular expression.
  """
  new_screen_output = RichTextLines(
      orig_screen_output.lines,
      font_attr_segs=copy.deepcopy(orig_screen_output.font_attr_segs),
      annotations=orig_screen_output.annotations)

  try:
    re_prog = re.compile(regex)
  except re.error:
    raise ValueError("Invalid regular expression: \"%s\"" % regex)

  regex_match_lines = []
  for i, line in enumerate(new_screen_output.lines):
    find_it = re_prog.finditer(line)

    match_segs = []
    for match in find_it:
      match_segs.append((match.start(), match.end(), font_attr))

    if match_segs:
      if i not in new_screen_output.font_attr_segs:
        new_screen_output.font_attr_segs[i] = match_segs
      else:
        new_screen_output.font_attr_segs[i].extend(match_segs)
        new_screen_output.font_attr_segs[i] = sorted(
            new_screen_output.font_attr_segs[i], key=lambda x: x[0])
      regex_match_lines.append(i)

  new_screen_output.annotations[REGEX_MATCH_LINES_KEY] = regex_match_lines
  return new_screen_output


def wrap_rich_text_lines(inp, cols):
  # [FUNCTION DOC]
  # Hard-wraps a RichTextLines to at most cols characters per line. inp must be a
  # RichTextLines and cols an integer, otherwise a value error is raised. Lines longer than
  # cols are cut into chunks; font attribute segments are split and clipped across chunks
  # (zero-length pieces dropped); annotations for a row move to the first wrapped row; non-row
  # annotation keys are copied. Returns a pair: the new RichTextLines, and a list giving, for
  # each original line, the index of its first line in the output. Input is not modified.
  """Wrap RichTextLines according to maximum number of columns.

  Produces a new RichTextLines object with the text lines, font_attr_segs and
  annotations properly wrapped. This ought to be used sparingly, as in most
  cases, command handlers producing RichTextLines outputs should know the
  screen/panel width via the screen_info kwarg and should produce properly
  length-limited lines in the output accordingly.

  Args:
    inp: Input RichTextLines object.
    cols: Number of columns, as an int.

  Returns:
    1) A new instance of RichTextLines, with line lengths limited to cols.
    2) A list of new (wrapped) line index. For example, if the original input
      consists of three lines and only the second line is wrapped, and it's
      wrapped into two lines, this return value will be: [0, 1, 3].
  Raises:
    ValueError: If inputs have invalid types.
  """

  new_line_indices = []

  if not isinstance(inp, RichTextLines):
    raise ValueError("Invalid type of input screen_output")

  if not isinstance(cols, int):
    raise ValueError("Invalid type of input cols")

  out = RichTextLines([])

  row_counter = 0  # Counter for new row index
  for i, line in enumerate(inp.lines):
    new_line_indices.append(out.num_lines())

    if i in inp.annotations:
      out.annotations[row_counter] = inp.annotations[i]

    if len(line) <= cols:
      # No wrapping.
      out.lines.append(line)
      if i in inp.font_attr_segs:
        out.font_attr_segs[row_counter] = inp.font_attr_segs[i]

      row_counter += 1
    else:
      # Wrap.
      wlines = []  # Wrapped lines.

      osegs = []
      if i in inp.font_attr_segs:
        osegs = inp.font_attr_segs[i]

      idx = 0
      while idx < len(line):
        if idx + cols > len(line):
          rlim = len(line)
        else:
          rlim = idx + cols

        wlines.append(line[idx:rlim])
        for seg in osegs:
          if (seg[0] < rlim) and (seg[1] >= idx):
            # Calculate left bound within wrapped line.
            if seg[0] >= idx:
              lb = seg[0] - idx
            else:
              lb = 0

            # Calculate right bound within wrapped line.
            if seg[1] < rlim:
              rb = seg[1] - idx
            else:
              rb = rlim - idx

            if rb > lb:  # Omit zero-length segments.
              wseg = (lb, rb, seg[2])
              if row_counter not in out.font_attr_segs:
                out.font_attr_segs[row_counter] = [wseg]
              else:
                out.font_attr_segs[row_counter].append(wseg)

        idx += cols
        row_counter += 1

      out.lines.extend(wlines)

  # Copy over keys of annotation that are not row indices.
  for key in inp.annotations:
    if not isinstance(key, int):
      out.annotations[key] = inp.annotations[key]

  return out, new_line_indices


class CommandHandlerRegistry:
  # [CLASS DOC]
  # Registry mapping command prefixes (and their aliases) to handler callables. Dispatches a
  # command to its handler and returns the handler's RichTextLines, converting unexpected
  # errors into error text. Keeps tables of handlers, alias-to-prefix, prefix-to-aliases and
  # help strings, plus optional help intro text. On construction it registers the built-in
  # "help" (alias "h") and "version" (alias "ver") commands.
  """Registry of command handlers for CLI.

  Handler methods (callables) for user commands can be registered with this
  class, which then is able to dispatch commands to the correct handlers and
  retrieve the RichTextLines output.

  For example, suppose you have the following handler defined:
    def echo(argv, screen_info=None):
      return RichTextLines(["arguments = %s" % " ".join(argv),
                            "screen_info = " + repr(screen_info)])

  you can register the handler with the command prefix "echo" and alias "e":
    registry = CommandHandlerRegistry()
    registry.register_command_handler("echo", echo,
        "Echo arguments, along with screen info", prefix_aliases=["e"])

  then to invoke this command handler with some arguments and screen_info, do:
    registry.dispatch_command("echo", ["foo", "bar"], screen_info={"cols": 80})

  or with the prefix alias:
    registry.dispatch_command("e", ["foo", "bar"], screen_info={"cols": 80})

  The call will return a RichTextLines object which can be rendered by a CLI.
  """

  HELP_COMMAND = "help"
  HELP_COMMAND_ALIASES = ["h"]
  VERSION_COMMAND = "version"
  VERSION_COMMAND_ALIASES = ["ver"]

  def __init__(self):
    # [FUNCTION DOC]
    # Creates empty handler, alias and help tables, clears the help intro, then registers the
    # default help and version handlers (with their aliases). Takes no inputs, returns nothing.
    # A dictionary from command prefix to handler.
    self._handlers = {}

    # A dictionary from prefix alias to prefix.
    self._alias_to_prefix = {}

    # A dictionary from prefix to aliases.
    self._prefix_to_aliases = {}

    # A dictionary from command prefix to help string.
    self._prefix_to_help = {}

    # Introductory text to help information.
    self._help_intro = None

    # Register a default handler for the command "help".
    self.register_command_handler(
        self.HELP_COMMAND,
        self._help_handler,
        "Print this help message.",
        prefix_aliases=self.HELP_COMMAND_ALIASES)

    # Register a default handler for the command "version".
    self.register_command_handler(
        self.VERSION_COMMAND,
        self._version_handler,
        "Print the versions of TensorFlow and its key dependencies.",
        prefix_aliases=self.VERSION_COMMAND_ALIASES)

  def register_command_handler(self,
                               prefix,
                               handler,
                               help_info,
                               prefix_aliases=None):
    # [FUNCTION DOC]
    # Registers handler (a callable taking an argument list and an optional screen info
    # mapping, returning a RichTextLines) for the command word prefix, with help_info (a string)
    # and optional prefix_aliases (list of strings). Raises a value error if the prefix is
    # empty, already registered, the handler is not callable, help_info is not a string, or an
    # alias clashes with an existing prefix or alias. Side effect: updates the registry tables.
    # Note: validation happens in order, and aliases are stored as they are checked, so a
    # clash part-way leaves earlier aliases of that call registered.
    """Register a callable as a command handler.

    Args:
      prefix: Command prefix, i.e., the first word in a command, e.g.,
        "print" as in "print tensor_1".
      handler: A callable of the following signature:
          foo_handler(argv, screen_info=None),
        where argv is the argument vector (excluding the command prefix) and
          screen_info is a dictionary containing information about the screen,
          such as number of columns, e.g., {"cols": 100}.
        The callable should return:
          1) a RichTextLines object representing the screen output.

        The callable can also raise an exception of the type CommandLineExit,
        which if caught by the command-line interface, will lead to its exit.
        The exception can optionally carry an exit token of arbitrary type.
      help_info: A help string.
      prefix_aliases: Aliases for the command prefix, as a list of str. E.g.,
        shorthands for the command prefix: ["p", "pr"]

    Raises:
      ValueError: If
        1) the prefix is empty, or
        2) handler is not callable, or
        3) a handler is already registered for the prefix, or
        4) elements in prefix_aliases clash with existing aliases.
        5) help_info is not a str.
    """

    if not prefix:
      raise ValueError("Empty command prefix")

    if prefix in self._handlers:
      raise ValueError(
          "A handler is already registered for command prefix \"%s\"" % prefix)

    # Make sure handler is callable.
    if not callable(handler):
      raise ValueError("handler is not callable")

    # Make sure that help info is a string.
    if not isinstance(help_info, str):
      raise ValueError("help_info is not a str")

    # Process prefix aliases.
    if prefix_aliases:
      for alias in prefix_aliases:
        if self._resolve_prefix(alias):
          raise ValueError(
              "The prefix alias \"%s\" clashes with existing prefixes or "
              "aliases." % alias)
        self._alias_to_prefix[alias] = prefix

      self._prefix_to_aliases[prefix] = prefix_aliases

    # Store handler.
    self._handlers[prefix] = handler

    # Store help info.
    self._prefix_to_help[prefix] = help_info

  def dispatch_command(self, prefix, argv, screen_info=None):
    # [FUNCTION DOC]
    # Runs the handler registered for prefix (or an alias) with argv (list of strings, without
    # the prefix) and optional screen_info mapping. Returns the handler's RichTextLines or null.
    # A command-line-exit exception is re-raised; a system-exit (argparse syntax error) becomes
    # a syntax-error text; any other exception becomes error text including type, message and
    # traceback. Raises a value error for an empty or unregistered prefix, or if the handler
    # returns something other than null or a RichTextLines.
    """Handles a command by dispatching it to a registered command handler.

    Args:
      prefix: Command prefix, as a str, e.g., "print".
      argv: Command argument vector, excluding the command prefix, represented
        as a list of str, e.g.,
        ["tensor_1"]
      screen_info: A dictionary containing screen info, e.g., {"cols": 100}.

    Returns:
      An instance of RichTextLines or None. If any exception is caught during
      the invocation of the command handler, the RichTextLines will wrap the
      error type and message.

    Raises:
      ValueError: If
        1) prefix is empty, or
        2) no command handler is registered for the command prefix, or
        3) the handler is found for the prefix, but it fails to return a
          RichTextLines or raise any exception.
      CommandLineExit:
        If the command handler raises this type of exception, this method will
        simply pass it along.
    """
    if not prefix:
      raise ValueError("Prefix is empty")

    resolved_prefix = self._resolve_prefix(prefix)
    if not resolved_prefix:
      raise ValueError("No handler is registered for command prefix \"%s\"" %
                       prefix)

    handler = self._handlers[resolved_prefix]
    try:
      output = handler(argv, screen_info=screen_info)
    except CommandLineExit as e:
      raise e
    except SystemExit as e:
      # Special case for syntax errors caught by argparse.
      lines = ["Syntax error for command: %s" % prefix,
               "For help, do \"help %s\"" % prefix]
      output = RichTextLines(lines)

    except BaseException as e:  # pylint: disable=broad-except
      lines = ["Error occurred during handling of command: %s %s:" %
               (resolved_prefix, " ".join(argv)), "%s: %s" % (type(e), str(e))]

      # Include traceback of the exception.
      lines.append("")
      lines.extend(traceback.format_exc().split("\n"))

      output = RichTextLines(lines)

    if not isinstance(output, RichTextLines) and output is not None:
      raise ValueError(
          "Return value from command handler %s is not None or a RichTextLines "
          "instance" % str(handler))

    return output

  def is_registered(self, prefix):
    # [FUNCTION DOC]
    # Returns true if prefix, as a command prefix or alias, resolves to a registered handler.
    """Test if a command prefix or its alias is has a registered handler.

    Args:
      prefix: A prefix or its alias, as a str.

    Returns:
      True iff a handler is registered for prefix.
    """
    return self._resolve_prefix(prefix) is not None

  def get_help(self, cmd_prefix=None):
    # [FUNCTION DOC]
    # Builds help text as a RichTextLines. With no cmd_prefix (null or empty) it returns the
    # help intro (if set) followed by the help of every command in sorted prefix order, each
    # followed by two blank lines. Otherwise it returns the help for the given prefix or alias
    # (an "invalid command prefix" message if unknown).
    """Compile help information into a RichTextLines object.

    Args:
      cmd_prefix: Optional command prefix. As the prefix itself or one of its
        aliases.

    Returns:
      A RichTextLines object containing the help information. If cmd_prefix
      is None, the return value will be the full command-line help. Otherwise,
      it will be the help information for the specified command.
    """
    if not cmd_prefix:
      # Print full help information, in sorted order of the command prefixes.
      help_info = RichTextLines([])
      if self._help_intro:
        # If help intro is available, show it at the beginning.
        help_info.extend(self._help_intro)

      sorted_prefixes = sorted(self._handlers)
      for cmd_prefix in sorted_prefixes:
        lines = self._get_help_for_command_prefix(cmd_prefix)
        lines.append("")
        lines.append("")
        help_info.extend(RichTextLines(lines))

      return help_info
    else:
      return RichTextLines(self._get_help_for_command_prefix(cmd_prefix))

  def set_help_intro(self, help_intro):
    # [FUNCTION DOC]
    # Stores help_intro (a RichTextLines) to be shown at the start of the full help output.
    # Replaces any previous intro. Returns nothing.
    """Set an introductory message to help output.

    Args:
      help_intro: (RichTextLines) Rich text lines appended to the
        beginning of the output of the command "help", as introductory
        information.
    """
    self._help_intro = help_intro

  def _help_handler(self, args, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: Built-in handler for the "help" command registered by the constructor.
    # args is the argument list after "help"; screen_info is ignored. With no arguments returns
    # the full help, with one argument the help of that command, and with more than one returns
    # an error line (not an exception). Returns a RichTextLines.
    """Command handler for "help".

    "help" is a common command that merits built-in support from this class.

    Args:
      args: Command line arguments to "help" (not including "help" itself).
      screen_info: (dict) Information regarding the screen, e.g., the screen
        width in characters: {"cols": 80}

    Returns:
      (RichTextLines) Screen text output.
    """

    _ = screen_info  # Unused currently.

    if not args:
      return self.get_help()
    elif len(args) == 1:
      return self.get_help(args[0])
    else:
      return RichTextLines(["ERROR: help takes only 0 or 1 input argument."])

  def _version_handler(self, args, screen_info=None):
    # [FUNCTION DOC]
    # HELPER: Built-in handler for the "version" command registered by the constructor.
    # Ignores args and screen_info and returns the TensorFlow version lines including
    # dependency (numpy) versions, as a RichTextLines.
    del args  # Unused currently.
    del screen_info  # Unused currently.
    return get_tensorflow_version_lines(include_dependency_versions=True)

  def _resolve_prefix(self, token):
    # [FUNCTION DOC]
    # HELPER: Alias lookup used by registration, dispatch and help.
    # Returns token if it is a registered prefix, the prefix an alias maps to, or null if
    # neither.
    """Resolve command prefix from the prefix itself or its alias.

    Args:
      token: a str to be resolved.

    Returns:
      If resolvable, the resolved command prefix.
      If not resolvable, None.
    """
    if token in self._handlers:
      return token
    elif token in self._alias_to_prefix:
      return self._alias_to_prefix[token]
    else:
      return None

  def _get_help_for_command_prefix(self, cmd_prefix):
    # [FUNCTION DOC]
    # HELPER: Produces help lines for a single command for the help output.
    # Returns a list of strings: the resolved prefix, an indented alias line if aliases exist,
    # a blank line, then the indented help text lines. For an unknown prefix returns a single
    # "invalid command prefix" line.
    """Compile the help information for a given command prefix.

    Args:
      cmd_prefix: Command prefix, as the prefix itself or one of its aliases.

    Returns:
      A list of str as the help information for cmd_prefix. If the cmd_prefix
        does not exist, the returned list of str will indicate that.
    """
    lines = []

    resolved_prefix = self._resolve_prefix(cmd_prefix)
    if not resolved_prefix:
      lines.append("Invalid command prefix: \"%s\"" % cmd_prefix)
      return lines

    lines.append(resolved_prefix)

    if resolved_prefix in self._prefix_to_aliases:
      lines.append(HELP_INDENT + "Aliases: " + ", ".join(
          self._prefix_to_aliases[resolved_prefix]))

    lines.append("")
    help_lines = self._prefix_to_help[resolved_prefix].split("\n")
    for line in help_lines:
      lines.append(HELP_INDENT + line)

    return lines


class TabCompletionRegistry:
  # [CLASS DOC]
  # Registry of tab-completion candidates keyed by context word (an already-typed word that
  # decides which candidates apply). Holds a mapping from context word to a sorted list of
  # completion items and supports registering, deregistering, extending, removing and querying.
  """Registry for tab completion responses."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates an empty registry (no contexts). Returns nothing.
    self._comp_dict = {}

  # TODO(cais): Rename method names with "comp" to "*completion*" to avoid
  # confusion.

  def register_tab_comp_context(self, context_words, comp_items):
    # [FUNCTION DOC]
    # Registers a completion context: for every word in context_words (a list of strings) the
    # candidates become comp_items (a list of strings), stored sorted. All context words share
    # one sorted list object. Overwrites existing entries for those words. Raises a type error
    # if either argument is not a list.
    """Register a tab-completion context.

    Register that, for each word in context_words, the potential tab-completions
    are the words in comp_items.

    A context word is a pre-existing, completed word in the command line that
    determines how tab-completion works for another, incomplete word in the same
    command line.
    Completion items consist of potential candidates for the incomplete word.

    To give a general example, a context word can be "drink", and the completion
    items can be ["coffee", "tea", "water"]

    Note: A context word can be empty, in which case the context is for the
     top-level commands.

    Args:
      context_words: A list of context words belonging to the context being
        registered. It is a list of str, instead of a single string, to support
        synonym words triggering the same tab-completion context, e.g.,
        both "drink" and the short-hand "dr" can trigger the same context.
      comp_items: A list of completion items, as a list of str.

    Raises:
      TypeError: if the input arguments are not all of the correct types.
    """

    if not isinstance(context_words, list):
      raise TypeError("Incorrect type in context_list: Expected list, got %s" %
                      type(context_words))

    if not isinstance(comp_items, list):
      raise TypeError("Incorrect type in comp_items: Expected list, got %s" %
                      type(comp_items))

    # Sort the completion items on registration, so that later during
    # get_completions calls, no sorting will be necessary.
    sorted_comp_items = sorted(comp_items)

    for context_word in context_words:
      self._comp_dict[context_word] = sorted_comp_items

  def deregister_context(self, context_words):
    # [FUNCTION DOC]
    # Removes the contexts for each word in context_words (list of strings). Checks all words
    # first and raises a key error, changing nothing, if any is unregistered. Returns nothing.
    """Deregister a list of context words.

    Args:
      context_words: A list of context words to deregister, as a list of str.

    Raises:
      KeyError: if there are word(s) in context_words that do not correspond
        to any registered contexts.
    """

    for context_word in context_words:
      if context_word not in self._comp_dict:
        raise KeyError("Cannot deregister unregistered context word \"%s\"" %
                       context_word)

    for context_word in context_words:
      del self._comp_dict[context_word]

  def extend_comp_items(self, context_word, new_comp_items):
    # [FUNCTION DOC]
    # Adds new_comp_items (list of strings) to the candidates of context_word and re-sorts.
    # Raises a key error if the word is unregistered. Because context words registered together
    # initially share one list, the in-place extension is seen by all of them, but the re-sort
    # then rebinds only this word to a new list, so they may diverge afterwards.
    """Add a list of completion items to a completion context.

    Args:
      context_word: A single completion word as a string. The extension will
        also apply to all other context words of the same context.
      new_comp_items: (list of str) New completion items to add.

    Raises:
      KeyError: if the context word has not been registered.
    """

    if context_word not in self._comp_dict:
      raise KeyError("Context word \"%s\" has not been registered" %
                     context_word)

    self._comp_dict[context_word].extend(new_comp_items)
    self._comp_dict[context_word] = sorted(self._comp_dict[context_word])

  def remove_comp_items(self, context_word, comp_items):
    # [FUNCTION DOC]
    # Removes each item in comp_items from the candidates of context_word, in place (so words
    # sharing the same list are affected too). Raises a key error if the word is unregistered
    # and a value error if an item is not present (earlier items stay removed). Returns nothing.
    """Remove a list of completion items from a completion context.

    Args:
      context_word: A single completion word as a string. The removal will
        also apply to all other context words of the same context.
      comp_items: Completion items to remove.

    Raises:
      KeyError: if the context word has not been registered.
    """

    if context_word not in self._comp_dict:
      raise KeyError("Context word \"%s\" has not been registered" %
                     context_word)

    for item in comp_items:
      self._comp_dict[context_word].remove(item)

  def get_completions(self, context_word, prefix):
    # [FUNCTION DOC]
    # Looks up candidates for context_word that start with prefix (string). Returns a pair:
    # the sorted list of matching items and their longest common prefix; both are null if the
    # context word is unregistered. The list may be empty (common prefix then empty string).
    """Get the tab completions given a context word and a prefix.

    Args:
      context_word: The context word.
      prefix: The prefix of the incomplete word.

    Returns:
      (1) None if no registered context matches the context_word.
          A list of str for the matching completion items. Can be an empty list
          of a matching context exists, but no completion item matches the
          prefix.
      (2) Common prefix of all the words in the first return value. If the
          first return value is None, this return value will be None, too. If
          the first return value is not None, i.e., a list, this return value
          will be a str, which can be an empty str if there is no common
          prefix among the items of the list.
    """

    if context_word not in self._comp_dict:
      return None, None

    comp_items = self._comp_dict[context_word]
    comp_items = sorted(
        [item for item in comp_items if item.startswith(prefix)])

    return comp_items, self._common_prefix(comp_items)

  def _common_prefix(self, m):
    # [FUNCTION DOC]
    # HELPER: Used by get_completions to compute the shared prefix of the matching items.
    # Takes a list of strings and returns their longest common prefix (empty string for an empty
    # list), found by comparing the smallest and largest strings.
    """Given a list of str, returns the longest common prefix.

    Args:
      m: (list of str) A list of strings.

    Returns:
      (str) The longest common prefix.
    """
    if not m:
      return ""

    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
      if c != s2[i]:
        return s1[:i]

    return s1


class CommandHistory:
  # [CLASS DOC]
  # Bounded list of recent commands, persisted to a history file (default name in the user's
  # home directory) so it survives sessions. Supports adding, fetching the most recent n and
  # prefix lookup. Holds the command list, the size limit and the history file path.
  """Keeps command history and supports lookup."""

  _HISTORY_FILE_NAME = ".tfdbg_history"

  def __init__(self, limit=100, history_file_path=None):
    # [FUNCTION DOC]
    # Creates the history. limit (int, default 100) is the maximum number of commands kept;
    # history_file_path (string, optional) overrides the default file in the home directory
    # (used in tests). Loads existing history from the file, which may trim and rewrite it.
    # Returns nothing.
    """CommandHistory constructor.

    Args:
      limit: Maximum number of the most recent commands that this instance
        keeps track of, as an int.
      history_file_path: (str) Manually specified path to history file. Used in
        testing.
    """

    self._commands = []
    self._limit = limit
    self._history_file_path = (
        history_file_path or self._get_default_history_file_path())
    self._load_history_from_file()

  def _load_history_from_file(self):
    # [FUNCTION DOC]
    # HELPER: Loads saved history during construction.
    # If the history file exists, reads its non-blank lines (stripped) as the command list; if
    # more than the limit, keeps only the newest ones and rewrites the file with them. On an
    # I/O error prints a warning instead of raising. Does nothing if the file is missing.
    if os.path.isfile(self._history_file_path):
      try:
        with open(self._history_file_path, "rt") as history_file:
          commands = history_file.readlines()
        self._commands = [command.strip() for command in commands
                          if command.strip()]

        # Limit the size of the history file.
        if len(self._commands) > self._limit:
          self._commands = self._commands[-self._limit:]
          with open(self._history_file_path, "wt") as history_file:
            for command in self._commands:
              history_file.write(command + "\n")
      except IOError:
        print("WARNING: writing history file failed.")

  def _add_command_to_history_file(self, command):
    # [FUNCTION DOC]
    # HELPER: Persists one command for add_command. Appends the command and a newline to the
    # history file; I/O errors are silently ignored. Returns nothing.
    try:
      with open(self._history_file_path, "at") as history_file:
        history_file.write(command + "\n")
    except IOError:
      pass

  @classmethod
  def _get_default_history_file_path(cls):
    # [FUNCTION DOC]
    # HELPER: Class method giving the default history file location: the history file name
    # inside the current user's home directory. Returns a path string.
    return os.path.join(os.path.expanduser("~"), cls._HISTORY_FILE_NAME)

  def add_command(self, command):
    # [FUNCTION DOC]
    # Records command (a string) as the newest history entry. A command equal to the latest
    # entry is ignored. Raises a type error if command is not a string. Trims the list to the
    # limit and appends the command to the history file (file errors ignored).
    # KNOWN ISSUE: the repeat check runs before the type check, so a non-string never raises
    # when it merely compares equal to the last entry (only an edge case).
    """Add a command to the command history.

    Args:
      command: The history command, as a str.

    Raises:
      TypeError: if command is not a str.
    """

    if self._commands and command == self._commands[-1]:
      # Ignore repeating commands in a row.
      return

    if not isinstance(command, str):
      raise TypeError("Attempt to enter non-str entry to command history")

    self._commands.append(command)

    if len(self._commands) > self._limit:
      self._commands = self._commands[-self._limit:]

    self._add_command_to_history_file(command)

  def most_recent_n(self, n):
    # [FUNCTION DOC]
    # Returns the newest n commands (int) in chronological order, or all if fewer exist.
    """Look up the n most recent commands.

    Args:
      n: Number of most recent commands to look up.

    Returns:
      A list of n most recent commands, or all available most recent commands,
      if n exceeds size of the command history, in chronological order.
    """

    return self._commands[-n:]

  def lookup_prefix(self, prefix, n):
    # [FUNCTION DOC]
    # Returns up to the newest n commands (chronological order) that start with prefix (string).
    """Look up the n most recent commands that starts with prefix.

    Args:
      prefix: The prefix to lookup.
      n: Number of most recent commands to look up.

    Returns:
      A list of n most recent commands that have the specified prefix, or all
      available most recent commands that have the prefix, if n exceeds the
      number of history commands with the prefix.
    """

    commands = [cmd for cmd in self._commands if cmd.startswith(prefix)]

    return commands[-n:]

  # TODO(cais): Lookup by regex.


class MenuItem:
  # [CLASS DOC]
  # One entry of a text menu: a caption shown to the user, content (for example a command
  # string to run when selected) and an enabled flag that can be toggled.
  """A class for an item in a text-based menu."""

  def __init__(self, caption, content, enabled=True):
    # [FUNCTION DOC]
    # Creates a menu item with caption (string), content (any object, e.g. a command string)
    # and enabled (bool, default true). Returns nothing.
    # KNOWN ISSUE: the docstring says "Menu constructor" but this builds a MenuItem.
    """Menu constructor.

    TODO(cais): Nested menu is currently not supported. Support it.

    Args:
      caption: (str) caption of the menu item.
      content: Content of the menu item. For a menu item that triggers
        a command, for example, content is the command string.
      enabled: (bool) whether this menu item is enabled.
    """

    self._caption = caption
    self._content = content
    self._enabled = enabled

  @property
  def caption(self):
    # [FUNCTION DOC]
    # Property returning the caption string.
    return self._caption

  @property
  def type(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: reads an attribute that is never set anywhere in this class, so accessing it
    # raises an attribute error (dead or leftover code).
    # Intended to return the node type of the item.
    return self._node_type

  @property
  def content(self):
    # [FUNCTION DOC]
    # Property returning the content object given at construction.
    return self._content

  def is_enabled(self):
    # [FUNCTION DOC]
    # Returns true if the item is currently enabled.
    return self._enabled

  def disable(self):
    # [FUNCTION DOC]
    # Marks the item as disabled in place. Returns nothing.
    self._enabled = False

  def enable(self):
    # [FUNCTION DOC]
    # Marks the item as enabled in place. Returns nothing.
    self._enabled = True


class Menu:
  # [CLASS DOC]
  # Ordered text-based menu made of MenuItem objects, with an optional name. Supports adding
  # items, lookup by caption and rendering as a single RichTextLines line.
  """A class for text-based menu."""

  def __init__(self, name=None):
    # [FUNCTION DOC]
    # Creates an empty menu with an optional name (string or null). Returns nothing.
    """Menu constructor.

    Args:
      name: (str or None) name of this menu.
    """

    self._name = name
    self._items = []

  def append(self, item):
    # [FUNCTION DOC]
    # Adds item (a MenuItem) at the end of the menu. Returns nothing.
    """Append an item to the Menu.

    Args:
      item: (MenuItem) the item to be appended.
    """
    self._items.append(item)

  def insert(self, index, item):
    # [FUNCTION DOC]
    # Inserts item (a MenuItem) at position index (int) in the menu. Returns nothing.
    self._items.insert(index, item)

  def num_items(self):
    # [FUNCTION DOC]
    # Returns how many items the menu holds.
    return len(self._items)

  def captions(self):
    # [FUNCTION DOC]
    # Returns the list of item captions in menu order.
    return [item.caption for item in self._items]

  def caption_to_item(self, caption):
    # [FUNCTION DOC]
    # Returns the first item whose caption equals caption (string). Raises a lookup error if
    # there is none.
    """Get a MenuItem from the caption.

    Args:
      caption: (str) The caption to look up.

    Returns:
      (MenuItem) The first-match menu item with the caption, if any.

    Raises:
      LookupError: If a menu item with the caption does not exist.
    """

    captions = self.captions()
    if caption not in captions:
      raise LookupError("There is no menu item with the caption \"%s\"" %
                        caption)

    return self._items[captions.index(caption)]

  def format_as_single_line(self,
                            prefix=None,
                            divider=" | ",
                            enabled_item_attrs=None,
                            disabled_item_attrs=None):
    # [FUNCTION DOC]
    # Renders the menu as one line of text, returned as a one-line RichTextLines. Inputs:
    # prefix (optional string put first), divider (string placed after every item, default a
    # bar with spaces), enabled_item_attrs and disabled_item_attrs (optional list or single
    # attribute; a single value is wrapped in a list). Enabled items get a segment whose
    # attribute list starts with the item object itself followed by the enabled attributes;
    # disabled items get a segment only if disabled attributes are given. The divider is
    # also appended after the last item.
    """Format the menu as a single-line RichTextLines object.

    Args:
      prefix: (str) String added to the beginning of the line.
      divider: (str) The dividing string between the menu items.
      enabled_item_attrs: (list or str) Attributes applied to each enabled
        menu item, e.g., ["bold", "underline"].
      disabled_item_attrs: (list or str) Attributes applied to each
        disabled menu item, e.g., ["red"].

    Returns:
      (RichTextLines) A single-line output representing the menu, with
        font_attr_segs marking the individual menu items.
    """

    if (enabled_item_attrs is not None and
        not isinstance(enabled_item_attrs, list)):
      enabled_item_attrs = [enabled_item_attrs]

    if (disabled_item_attrs is not None and
        not isinstance(disabled_item_attrs, list)):
      disabled_item_attrs = [disabled_item_attrs]

    menu_line = prefix if prefix is not None else ""
    attr_segs = []

    for item in self._items:
      menu_line += item.caption
      item_name_begin = len(menu_line) - len(item.caption)

      if item.is_enabled():
        final_attrs = [item]
        if enabled_item_attrs:
          final_attrs.extend(enabled_item_attrs)
        attr_segs.append((item_name_begin, len(menu_line), final_attrs))
      else:
        if disabled_item_attrs:
          attr_segs.append(
              (item_name_begin, len(menu_line), disabled_item_attrs))

      menu_line += divider

    return RichTextLines(menu_line, font_attr_segs={0: attr_segs})
