# [MODULE DOC]
# Command-line tool (and importable library) that rewrites an HLO text test file so that every
# test case is preceded by automatically generated FileCheck expectations. For each test case
# (cases are separated by lines made of a double slash followed by five or more dashes) it strips
# existing check/run directives, runs a user-supplied HLO optimizer program on the case through a
# temporary file, converts the optimizer's textual output into FileCheck directive comments
# (label check for module and function declarations, next-line checks for instructions, with
# instruction and function names replaced by regular expression captures), and writes the
# directives followed by the original case. Run directives found in the cases are moved to the top
# of the output, below a banner comment saying the assertions were generated. Test cases may be
# processed in parallel by a thread pool. Output goes to standard output, or the input file is
# overwritten when in-place mode is requested; the file name "-" means standard input/output.
# Contents: the directive comment enumeration, helper functions for whitespace and function
# declaration checks, the optimizer runner, a stream splitting helper, the check line generator,
# the test case writer class, argument parsing, and the entry point. No import-time side effects
# other than computing module constants (script name, banner, regular expressions).
# Standalone tool; not a helper module, although the entire logic is reusable by tests.
#!/usr/bin/env python3
# Copyright 2025 The OpenXLA Authors.
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
r"""Transforms an HLO test file, inserting FileCheck directives above each test.

This tool runs an HLO optimizer on each test case within a .hlo file, converts
the optimized HLO into FileCheck directives, and inserts the directives into the
test file above the corresponding test case.

Usage:
  ./generate_hlo_test_checks.py TEST_FILE [FLAGS...] -- OPT_CMD [OPT_ARGS...]

Args:
  TEST_FILE:
    The path to the input `.hlo` test file, or "-" to read from `stdin`.
  FLAGS...:
    Optional flags consumed by this tool itself.
      -h, --help:
        Print help and exit.
      -i, --in_place:
        Modify `TEST_FILE` in place instead of printing the results to stdout.
        (A `TEST_FILE` argument of "-" overrides this flag.)
      -I EXPAND, --expand_to_input EXPAND:
        Each instance of `EXPAND` in `OPT_ARGS...` will expand to the path of a
        temporary input file. `OPT_CMD` will be invoked separately on each test
        case, and this placeholder will expand to a different path each time.
        Defaults to "{}".
  --:
    A literal "--" string, acting as a sentinel token to separate this tool's
    own arguments from the optimization command.
  OPT_CMD:
    The HLO optimizer program to run, e.g. "hlo-opt". (Qualify the name with a
    path if appropriate, as though invoking it directly from the command line.)
  OPT_ARGS...:
    Arguments that will be forwarded to `OPT_CMD`, with one caveat: wherever you
    would normally specify an input-file path, simply use a literal "{}" string.
    This behaves much like `xargs` with the `-I{}` flag.
    NOTE: It's important to use "{}" here instead of repeating the `TEST_FILE`
    argument; the latter is not only redundant, but also incorrect.
    NOTE: If your optimizer command conflicts with the expansion token, you can
    specify a different expansion token with the `-I` flag; see above.
    NOTE: The `--split-input-file` flag (or equivalent) is unnecessary; this
    tool automatically splits the input file at "// -----" lines, so the
    optimizer will always be run on single-test-case input files.

Example Commands:
  Basic usage example: print the transformed test file to `stdout`.
    ```
    ./generate_hlo_test_checks.py tests/my_pass.hlo \
        -- hlo-opt {} --passes=my_pass
    ```
    TEST_FILE:   tests/my_pass.hlo
    FLAGS...:
    OPT_CMD:     hlo-opt
    OPT_ARGS...: {} --passes=my_pass

  Modify `/path/to/my_pass_test.hlo` in place (`-i`); use a custom expansion
  placeholder string of "%s" (`-I%s`) instead of the default "{}".
    ```
    ./generate_hlo_test_checks.py /path/to/my_pass_test.hlo -i -I%s \
        -- /path/to/hlo-opt %s --passes=my_pass
    ```
    TEST_FILE:   /path/to/my_pass_test.hlo
    FLAGS...:    -i -I%s
    OPT_CMD:     /path/to/hlo-opt
    OPT_ARGS...: %s --passes=my_pass

  Read from `stdin`, and write to `stdout` (indicated by `TEST_FILE` being "-").
    ```
    cat /path/to/foo_bar.hlo \
      | ./generate_hlo_test_checks.py - -- tools/hlo-opt {} --passes=foo,bar
    ```
    TEST_FILE:   -
    FLAGS...:
    OPT_CMD:     tools/hlo-opt
    OPT_ARGS...: {} --passes=foo,bar
"""

from __future__ import annotations
import argparse
import collections
from collections.abc import Callable, Iterator
import enum
import functools
import io
import itertools
import multiprocessing.pool
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Generic, Optional, TypeVar, Union, cast


_T = TypeVar("_T")
ListOrTuple = Union[list[_T], tuple[_T, ...]]

_SCRIPT_NAME: str = os.path.basename(__file__)
_BANNER_COMMENT_LINE: str = (
    f"// NOTE: Assertions have been autogenerated by hlo/tools/{_SCRIPT_NAME}\n"
)

_STANDARD_IO_STREAMS: str = "-"
_DEFAULT_INPUT_FILE_EXPANSION_TOKEN: str = "{}"

_FUNCTION_DECLARATION_REGEX: re.Pattern[str] = re.compile(
    r"^((?:[\w]+ )*)%([\w\-]+(?:\.[\w\-]+)*?(\.\d+)?)(?= [({])"
)

_DIRECTIVE_REGEX_MATCHER: re.Pattern[str] = re.compile(
    r"^ *// *(CHECK(?:-(?:COUNT|DAG|EMPTY|LABEL|NEXT|NOT|SAME))?|COM|RUN):")
_DIRECTIVE_MAX_STRING_LENGTH: int = len("CHECK-LABEL")


class DirectiveComment(enum.Enum):
  # [CLASS DOC]
  # Enumeration of the LLVM test directive comment kinds that this tool recognizes: the check
  # directive and its variants (count, dag, empty, label, next, not, same), the comment directive
  # and the run directive. Member names use underscores where the textual directive uses dashes.
  # Provides parsing from text, detection of the directive on a line of text, and formatting of
  # a line that carries the directive. Used by the stream splitter and the check generator.
  """LLVM directive comments.

  See relevant LLVM documentation pages:
  - https://llvm.org/docs/TestingGuide.html
  - https://llvm.org/docs/CommandGuide/FileCheck.html
  """

  CHECK = 0
  CHECK_COUNT = 1
  CHECK_DAG = 2
  CHECK_EMPTY = 3
  CHECK_LABEL = 4
  CHECK_NEXT = 5
  CHECK_NOT = 6
  CHECK_SAME = 7
  COM = 8
  RUN = 9

  def __str__(self) -> str:
    # [FUNCTION DOC]
    # Returns the textual form of the directive as it appears in test files: the member name with
    # underscores replaced by dashes (for example the label variant of the check directive).
    # No side effects.
    # Note: `DirectiveComment.padding_width` makes use of the fact that
    # `len(str(self)) == len(self.name)` to avoid an unnecessary `str.replace()`
    # operation. If that assumption ever ceases to hold in all cases, update
    # `padding_width` to call `len(str(self))` instead of `len(self.name)`.
    return self.name.replace("_", "-")

  @property
  def is_check(self) -> bool:
    # [FUNCTION DOC]
    # Property: true when this member is one of the check directive variants (name starts with the
    # word CHECK), false for the comment and run directives. No side effects.
    return self.name.startswith("CHECK")

  @property
  def padding_width(self) -> int:
    # [FUNCTION DOC]
    # Property: number of padding spaces needed after the directive so that the text of all check
    # variants lines up. It is the length of the longest check name minus the length of this
    # member name for check directives, and zero for the comment and run directives.
    # Relies on the textual form having the same length as the member name. No side effects.
    # Note: This uses the fact that `len(str(self)) == len(self.name)` to avoid
    # an unnecessary `str.replace()` operation. If that assumption ever ceases
    # to hold in all cases, replace `len(self.name)` with `len(str(self))`.
    return _DIRECTIVE_MAX_STRING_LENGTH - len(self.name) if self.is_check else 0

  @property
  def line_prefix(self) -> str:
    # [FUNCTION DOC]
    # Property: the text placed at the start of a generated line for this directive: a double
    # slash comment marker, the directive text, a colon, a space, and the alignment padding.
    # No side effects.
    return f"// {self}: {' ' * self.padding_width}"

  @classmethod
  def parse(cls, check_string: str) -> DirectiveComment:
    # [FUNCTION DOC]
    # Class method. Converts the textual directive name (dashes allowed in place of underscores,
    # e.g. the label variant of the check directive) into the corresponding enumeration member.
    # Input: check_string, the directive name without a colon. Returns the member. Raises a key
    # error if the name is not a known directive. No side effects.
    """Parses a string representation of a DirectiveComment."""
    return cls[check_string.replace("-", "_")]

  @classmethod
  def extract_from_line(cls, line: str) -> Optional[DirectiveComment]:
    # [FUNCTION DOC]
    # Class method. Detects whether a line of text is a comment starting with one of the known
    # directives (optional leading spaces, a double slash, optional spaces, the directive and a
    # colon). Input: line, one line of text. Returns the directive member, or the null value when
    # the line does not start with a directive. No side effects.
    """Returns the FileCheck/RUN directive, if any, used by a line of text."""
    match = _DIRECTIVE_REGEX_MATCHER.match(line)
    return None if match is None else cls.parse(match.group(1))

  def format_line(self, line_text: str) -> str:
    # [FUNCTION DOC]
    # Builds a complete test file line for this directive. Input: line_text, the text to place after
    # the directive. Returns the aligned directive prefix followed by the text and a newline.
    # No side effects.
    """Formats a comment containing this directive and the specified text."""
    return f"{self.line_prefix}{line_text}\n"


def _format_function_declaration_file_check(line_text: str) -> str:
  # [FUNCTION DOC]
  # HELPER: creates the check line for a function (or module section header) declaration line
  # when generating expectations in HloFileCheckLines.
  # Input: line_text, the text of the first line of a section, expected to start with optional
  # keyword words followed by a percent sign and a function name and then a space and an opening
  # parenthesis or brace. Returns a check line: the label variant of the directive when the name
  # has no numeric suffix after a final dot (an explicit symbol name to be matched verbatim),
  # otherwise the plain check variant (a numbered auto-generated name). The keywords and name
  # are kept; the remainder of the line is dropped. Raises a value error if the line does not
  # match the expected declaration pattern. No other side effects.
  """Returns a FileCheck line checking for a function declaration."""
  match = _FUNCTION_DECLARATION_REGEX.match(line_text)

  if match is None:
    raise ValueError(f"The line {repr(line_text)} did not match the expected "
                     f"regex pattern for a function declaration.")

  prefix_keywords = match.group(1)
  function_name = match.group(2)
  optional_line_number_suffix = match.group(3)

  function_name_line: str = f"{prefix_keywords}%{function_name}"

  treat_function_name_as_explicit_symbol: bool = (
      optional_line_number_suffix is None
  )

  directive = (
      DirectiveComment.CHECK_LABEL
      if treat_function_name_as_explicit_symbol
      else DirectiveComment.CHECK
  )

  return directive.format_line(function_name_line)


def optimize_hlo(
    input_stream: Iterator[str],
    optimizer_path: str,
    optimizer_args: ListOrTuple[str],
    expand_to_input: str = _DEFAULT_INPUT_FILE_EXPANSION_TOKEN,
) -> Iterator[str]:
  # [FUNCTION DOC]
  # Runs an external HLO optimizer program on one test case and returns its standard output.
  # Inputs: input_stream, an iterator over lines of the test case; optimizer_path, the program to
  # run; optimizer_args, list or tuple of arguments, in which every occurrence of the expansion
  # token is replaced with the path of a temporary file holding the test case; expand_to_input,
  # the token (default is the two-character brace placeholder).
  # Returns an in-memory text stream with the decoded standard output of the program (despite the
  # declared iterator type, a text stream object, iterable by line). Runs synchronously and
  # eagerly: the process has finished when the function returns.
  # Side effects: creates and removes a temporary file with the suffix hlo, starts a subprocess
  # with captured output. Raises ChildProcessError if the program exits with a nonzero status;
  # the message holds the full command, the program's standard error indented, and the input file
  # contents with line numbers (or a note that the file is empty).
  """Passes `input_stream` through an optimizer; output is another stream.

  Args:
    input_stream: An iterator over the lines of HLO to run through an optimizer.
    optimizer_path: The program to use for optimizing the HLO.
    optimizer_args: The arguments to pass into the optimizer tool. All instances
      of the substring `expand_to_input` in the argument list will expand to the
      path of a temporary file containing the text of `input_stream`.
    expand_to_input: All instances of this substring in `optimizer_args` will
      expand to the path of a temporary file containing the text of
      `input_stream`.

  Returns:
    An iterator over the lines of HLO produced by the optimizer.

  Raises:
    ChildProcessError: If the optimizer exits with error status.
  """
  with tempfile.NamedTemporaryFile(mode="w+", suffix=".hlo") as test_case_file:
    test_case_file.writelines(input_stream)
    test_case_file.seek(0)

    command_argv = [
        optimizer_path,
        *(
            arg.replace(expand_to_input, test_case_file.name)
            for arg in optimizer_args
        ),
    ]

    hlo_opt_process = subprocess.run(
        command_argv,
        capture_output=True,
        check=False,  # Use custom error handling.
    )

    if hlo_opt_process.returncode != 0:
      line_number_prefix_width = len(str(sum(1 for _ in test_case_file)))
      test_case_file.seek(0)

      error_lines = list()

      error_lines.append(f"Error in command '{' '.join(command_argv)}':\n\n")
      for line in io.StringIO(hlo_opt_process.stderr.decode()):
        error_lines.append(f"  {line}")

      error_lines.append(
          f"\nContents of input file '{test_case_file.name}' "
          f"(expanded from '{expand_to_input}'), with line numbers:\n\n"
      )
      line_number = 0
      for line_number, line in enumerate(test_case_file, start=1):
        error_lines.append(
            f"  {line_number:>{line_number_prefix_width}d} {line}"
        )
      if line_number == 0:
        error_lines.append("  (File is empty.)\n")

      raise ChildProcessError("".join(error_lines))

  return io.StringIO(hlo_opt_process.stdout.decode())


class IterateByCategory(Generic[_T]):
  # [CLASS DOC]
  # Generic helper that lets several consumers read different sub-streams out of one input
  # iterator. A selector function, called on each item, decides which buffer (a double-ended
  # queue), or tuple of buffers, receives the item, or the null value to discard it. Consumers
  # ask for the next item of a particular buffer; items belonging to other buffers are stashed in
  # those buffers while reading ahead in the input. State: the input iterator and the selector.
  # Not thread safe. Used by HloStreamSplitter.
  """Sorts each element of an input stream into arbitrarily many output streams.
  """

  def __init__(
      self,
      input_stream: Iterator[_T],
      select_buffer: Callable[
          [_T],
          Union[collections.deque[_T], tuple[collections.deque[_T], ...], None],
      ],
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: input_stream, the iterator of items to categorize; select_buffer, a
    # callback taking an item and returning one buffer, a tuple of buffers, or the null value
    # (drop the item). Stores both; reads nothing yet. Returns nothing.
    self._input_stream: Iterator[_T] = input_stream
    self._select_buffer: Callable[
        [_T],
        Union[collections.deque[_T], tuple[collections.deque[_T], ...], None],
    ] = select_buffer

  def next_in_buffer(self, target_buffer: collections.deque[_T]) -> _T:
    # [FUNCTION DOC]
    # Returns the next item belonging to one particular buffer. If that buffer is not empty its
    # oldest item is removed and returned. Otherwise the input is read forward: dropped items are
    # skipped, items for other buffers are appended to those buffers, and the first item that
    # belongs to the target buffer is returned (an item selected for several buffers, as a tuple,
    # is also appended to the other buffers of the tuple).
    # Input: target_buffer, the queue identifying the sub-stream. Raises StopIteration when the
    # buffer and the input are both exhausted, and a type error if the selector returns something
    # other than a queue, a tuple of queues or the null value. Side effect: consumes the input
    # and fills other buffers.
    """Returns the next item in the sub-stream corresponding to `target_buffer`.

    Args:
      target_buffer: The queue backing the sub-stream whose next element should
        be returned.

    Returns:
      If `target_buffer` is nonempty, the next element of `target_buffer`.
      Otherwise, the next element of `self._input_stream` that would have been
      added to `target_buffer`.

    Raises:
      StopIteration: If `target_buffer` and `self._input_stream` are both empty.
    """
    if bool(target_buffer):
      return target_buffer.popleft()

    for item in self._input_stream:
      which_buffer = self._select_buffer(item)

      if which_buffer is None:
        continue
      if which_buffer is target_buffer:
        return item

      if isinstance(which_buffer, collections.deque):
        which_buffer.append(item)
        continue

      if isinstance(which_buffer, tuple):
        return_item: bool = False
        for buffer in which_buffer:
          if buffer is target_buffer:
            return_item = True
          else:
            buffer.append(item)
        if return_item:
          return item
        continue

      T = TypeVar("T", bound=_T)
      ExpectedTypes = Union[
          collections.deque[T], tuple[collections.deque[T], ...], None
      ]
      raise TypeError(
          f"`{self._select_buffer}` returned a value of type "
          f"`{type(which_buffer).__name__}`; expected one of `{ExpectedTypes}`."
      )

    raise StopIteration()

  def iterate_over_buffer(
      self, target_buffer: collections.deque[_T]
  ) -> Iterator[_T]:
    # [FUNCTION DOC]
    # Generator over one sub-stream. Input: target_buffer, the queue identifying the sub-stream.
    # Yields items of that sub-stream one at a time (reading ahead in the input as needed) until
    # the next-item function signals exhaustion, then ends. Side effects as for next_in_buffer.
    while True:
      try:
        yield self.next_in_buffer(target_buffer)
      except StopIteration:
        return


class HloStreamSplitter:
  # [CLASS DOC]
  # Removes check and run directive lines from a stream of test file lines. Lines that are not
  # directives (excluding the generated banner comment line, which is dropped to avoid
  # duplicating it) form the "non-directive" stream. Directive kinds listed at construction are
  # also recorded and can be read back as separate streams; other directives are discarded.
  # Built on IterateByCategory so the sub-streams can be consumed independently.
  """Splits out CHECK/RUN directives into separate streams.

  Strips away CHECK/RUN directives, optionally splitting them into their own
  streams if indicated by the `record_directives` constructor argument.
  """

  def __init__(self,
               input_stream: Iterator[str],
               record_directives: Optional[set[DirectiveComment]] = None):
    # [FUNCTION DOC]
    # Constructor. Inputs: input_stream, iterator of text lines; record_directives, optional set
    # of directive kinds whose lines should be kept in their own streams (default none, meaning
    # all directives are simply discarded). Creates an empty queue for the non-directive lines
    # and one queue per recorded directive, and wires them to a category iterator using the
    # selector method. Reads nothing yet. Returns nothing.
    if record_directives is None:
      record_directives = set()

    self._non_directive_lines: collections.deque[str] = collections.deque()

    self._directive_histories: dict[
        DirectiveComment, collections.deque[str]
    ] = {directive: collections.deque() for directive in record_directives}

    self._splitter: IterateByCategory[str] = IterateByCategory(
        input_stream,
        self._select_buffer,
    )

  def non_directive_lines(self) -> Iterator[str]:
    # [FUNCTION DOC]
    # Returns an iterator over the lines that are not directives (and are not the generated banner
    # comment), in original order. Consuming it may buffer directive lines read ahead.
    return self._splitter.iterate_over_buffer(self._non_directive_lines)

  def directive_history(self, directive: DirectiveComment) -> Iterator[str]:
    # [FUNCTION DOC]
    # Returns an iterator over the lines of one recorded directive kind, in original order, with
    # each line complete (including the directive text). Input: directive, a directive member that
    # was listed in record_directives at construction. Raises a key error with an explanatory
    # message if it was not. Consuming it may buffer other lines read ahead.
    try:
      directive_history_lines = self._directive_histories[directive]
    except KeyError as e:
      raise KeyError(
          f"This `HloStreamSplitter` was not configured to keep a record of "
          f'stripped "{directive}:" directives.'
      ) from e

    return self._splitter.iterate_over_buffer(directive_history_lines)

  def _select_buffer(
      self, line: str
  ) -> Union[collections.deque[str], tuple[collections.deque[str], ...], None]:
    # [FUNCTION DOC]
    # HELPER: selector callback given to the category iterator. Input: line, one text line.
    # Returns the queue of non-directive lines for ordinary lines, the null value (drop) for the
    # banner comment line or for a directive kind that is not being recorded, and the queue of
    # recorded lines for a recorded directive kind. No side effects.
    directive = DirectiveComment.extract_from_line(line)

    if directive is None:
      # Strip the top-of-file comment if present so as to avoid duplicating it.
      return None if line == _BANNER_COMMENT_LINE else self._non_directive_lines

    return self._directive_histories.get(directive, None)


def _fix_whitespace(input_stream: Iterator[str]) -> Iterator[str]:
  # [FUNCTION DOC]
  # HELPER: normalizes the text of a test case before it is sent to the optimizer and echoed back.
  # Input: input_stream, iterator of text lines. Yields the lines with all blank (whitespace-only)
  # lines removed, a single blank line inserted between sections (sections were separated by one
  # or more blank lines), no leading blank line, and every line indented by at least four spaces
  # re-indented to exactly two spaces (other lines unchanged). Lazy generator, no side effects.
  """Fixes indentation; trims redundant empty lines."""
  before_first_section: bool = True
  at_section_boundary: bool = True

  for line in input_stream:
    if not line.strip():
      at_section_boundary = True
      continue

    if at_section_boundary:
      if before_first_section:
        before_first_section = False
      else:
        yield "\n"
      at_section_boundary = False

    yield f"  {line.lstrip()}" if line.startswith("    ") else line


class HloFileCheckLines:
  # [CLASS DOC]
  # Converts the textual HLO output of an optimizer into FileCheck directive lines. It is an
  # iterable wrapper over an input line iterator: iterating yields the check lines. Pipeline: (1)
  # prefix each non-empty line with a check directive (label check for the module line, a
  # function-style check for the first line of each following section, next-line checks for the
  # rest), turning closing braces into an end-of-scope marker; (2) replace symbol names (names
  # after a percent sign) by regular expression captures, remembering earlier captures so later
  # references use the captured value; local symbols are forgotten at the end of each function
  # while function names persist. Label checks keep names verbatim. Class constants hold the
  # regular expressions and the marker line. State: input stream, section flags, a counter of
  # normalized names used for disambiguation, and local and global replacement caches.
  # Single-use: the instance holds state and should be iterated once.
  """Generates FileCheck comments from HLO IR."""

  _MODULE_REGEX: re.Pattern[str] = re.compile(
      r"^HloModule\b",
  )
  _CHECK_LINE_REGEX: re.Pattern[str] = re.compile(
      r"^// (CHECK(?:-\w+)?): .*?%[\w.\-]+ *(=|$)",
  )
  _SYMBOL_NAME_REGEX: re.Pattern[str] = re.compile(
      r"(?<=%)[\w\-]+(?:\.[\w\-]+)*(?:\.\d+)?",
  )
  _NON_SYMBOL_NAME_CHARS_REGEX: re.Pattern[str] = re.compile(
      r"\W",
  )

  _END_OF_FUNCTION_SCOPE_SENTINEL_VALUE: str = DirectiveComment.COM.format_line(
      "(End of function scope.)"
  )

  def __init__(self, input_stream: Iterator[str]):
    # [FUNCTION DOC]
    # Constructor. Input: input_stream, iterator over the lines of optimized HLO text. Stores it
    # and initializes the bookkeeping: first-line and section-break flags (both true), the table
    # counting how often each normalized name was used, and empty global and local symbol
    # replacement caches. Reads nothing yet. Returns nothing.
    """HloFileCheckLines constructor."""
    self._input_stream: Iterator[str] = input_stream
    self._on_first_line: bool = True
    self._at_section_break: bool = True
    self._num_symbols_with_normalized_name: dict[str, int] = dict()
    self._global_symbol_replacement_cache: dict[str, str] = dict()
    self._local_symbol_replacement_cache: dict[str, str] = dict()

  def __iter__(self) -> Iterator[str]:
    # [FUNCTION DOC]
    # Returns an iterator (generator) over the generated FileCheck lines: the input is first
    # prefixed with check directives and then has its symbol names replaced by captures. Lazy;
    # consumes the input stream and mutates the instance's caches and counters, so iterating twice
    # does not restart cleanly.
    """Converts HLO instructions to FileCheck directives where applicable."""
    return self._replace_symbol_names_with_regex_captures(
        self._prefix_lines_with_check_directives(self._input_stream)
    )

  def _prefix_lines_with_check_directives(
      self, input_stream: Iterator[str]
  ) -> Iterator[str]:
    # [FUNCTION DOC]
    # HELPER: first stage of the check generation pipeline. Input: input_stream, iterator over
    # lines of HLO text. Yields lines: blank lines are dropped and multiple ones collapse to one
    # blank separator between sections (none at the very start); a line that is only a closing
    # brace becomes the end-of-function-scope marker (a comment directive line); the module header
    # line becomes a label check; the first line of any other section becomes a function
    # declaration check (via the module-level declaration helper, which raises a value error if the
    # line does not look like a declaration); all other lines become next-line checks.
    # Mutates the section flags of the instance. Lazy generator.
    """Prepends "// CHECK-XXX:" directives to HLO instructions."""
    for line in input_stream:
      stripped_line = line.strip()

      # Keep track of section breaks, i.e. empty lines. If we see multiple empty
      # lines in a row, collapse them down into a single one. Also prune any
      # empty lines at the very start of the file.
      if not stripped_line:
        self._at_section_break = True
        continue

      # Leave out closing-brace lines, but replace them with "// COM:" (comment)
      # directives to tell the symbol replacer to clear its cache of local-scope
      # symbols. (The symbol replacer will also remove these added lines.)
      #
      # NOTE: We could just clear the local-symbol cache here instead of telling
      # the next stage of the pipeline to do it, but that would blur the API
      # boundaries and could introduce bugs if the iteration behavior changed.
      if stripped_line == "}":
        yield self._END_OF_FUNCTION_SCOPE_SENTINEL_VALUE
        continue

      first_line_of_new_section = self._at_section_break

      if self._at_section_break:
        if self._on_first_line:
          self._on_first_line = False
        else:
          yield "\n"
        self._at_section_break = False

      if self._MODULE_REGEX.match(stripped_line):
        yield DirectiveComment.CHECK_LABEL.format_line(stripped_line)
      elif first_line_of_new_section:
        yield _format_function_declaration_file_check(stripped_line)
      else:
        yield DirectiveComment.CHECK_NEXT.format_line(stripped_line)

  def _replace_symbol_names_with_regex_captures(
      self,
      input_stream: Iterator[str],
  ) -> Iterator[str]:
    # [FUNCTION DOC]
    # HELPER: second stage of the check generation pipeline. Input: input_stream, iterator over
    # lines (check lines produced by the first stage). Yields the lines with symbol names inside
    # check lines that contain an assignment or function declaration replaced by captures, via the
    # replacer callback; other lines pass through unchanged. The end-of-scope marker line is
    # swallowed (not yielded) and clears the local symbol cache. Label checks are treated as
    # verbatim and lines without an assignment sign (declarations) as global scope.
    # Mutates the replacement caches. Lazy generator.
    """Replaces HLO instruction & function names with FileCheck regex captures.

    Replaces explicit symbol names in FileCheck directives with regex
    captures. Lines that don't start with FileCheck directives are unchanged.

    Args:
      input_stream: An iterator to the lines of an HLO test.

    Yields:
      The transformed lines of the HLO test.
    """
    for line in input_stream:
      match: Optional[re.Match[str]] = self._CHECK_LINE_REGEX.match(line)

      if match is None:
        if line == self._END_OF_FUNCTION_SCOPE_SENTINEL_VALUE:
          self._local_symbol_replacement_cache.clear()
        else:
          yield line
        continue

      # "CHECK-LABEL" doesn't support regex captures; it's intended for symbols
      # with explicit names that should be checked verbatim.
      is_verbatim: bool = match.group(1) == "CHECK-LABEL"

      # `match.group(2)` captures "=" when matching an assignment and "" when
      # matching a function declaration. Functions should be treated as having
      # global scope, whereas assignments should go out of scope at the end of
      # a function.
      assert match.group(2) == "=" or not match.group(2)
      is_global: bool = not match.group(2)

      yield re.sub(self._SYMBOL_NAME_REGEX,
                   functools.partial(self._replacer,
                                     is_verbatim=is_verbatim,
                                     is_global=is_global),
                   line)

  def _replacer(
      self,
      match: re.Match[str],
      is_verbatim: bool,
      is_global: bool,
  ) -> str:
    # [FUNCTION DOC]
    # HELPER: substitution callback used with regular expression replacement in the second
    # pipeline stage. Inputs: match, the match for one symbol name; is_verbatim, whether the name
    # sits in a label check and must be kept as written; is_global, whether the name is a function
    # (global scope, remembered for the whole file) rather than an instruction (local scope).
    # Returns the replacement text. A name already seen in the local cache, or else the global
    # cache, is replaced by its stored reference (a capture reference). For a first occurrence the
    # function returns the declaring capture (a double-bracket capture of any run of non-space
    # characters, with a dollar marker for globals, named from a unique version of the symbol) and
    # stores the reference form for later uses; verbatim names are stored and returned unchanged.
    # Side effects: updates the local or global cache and the name counter.
    """A symbol-name replacement function for use in `re.sub`.

    Args:
      match: The match object produced by `self._SYMBOL_NAME_REGEX`.
      is_verbatim: Whether the newly matched symbol appears in a "CHECK-LABEL"
        directive, in which case it should be checked verbatim (not replaced
        with a regex capture).
      is_global: Whether the newly matched symbol appears in a declaration at
        global scope, i.e. whether it's a function name. If so, it should be
        remembered across function boundaries.

    Returns:
      The replacement string for the symbol name.
    """
    symbol_name = match.group(0)

    if symbol_name in self._local_symbol_replacement_cache:
      return self._local_symbol_replacement_cache[symbol_name]
    if symbol_name in self._global_symbol_replacement_cache:
      return self._global_symbol_replacement_cache[symbol_name]

    if is_verbatim:
      declaration_replacement = symbol_name
      reference_replacement = symbol_name
    else:
      capture_name = self._generate_unique_name(symbol_name)
      capture_pattern = r"[^ ]+"
      maybe_global_flag = "$" if is_global else ""
      declaration_replacement = (
          f"[[{maybe_global_flag}{capture_name}:{capture_pattern}]]"
      )
      reference_replacement = f"[[{maybe_global_flag}{capture_name}]]"

    if is_global:
      self._global_symbol_replacement_cache[symbol_name] = reference_replacement
    else:
      self._local_symbol_replacement_cache[symbol_name] = reference_replacement

    return declaration_replacement

  def _generate_unique_name(self, symbol_name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: gives each replaced symbol a distinct capture variable name for _replacer.
    # KNOWN ISSUE: uniqueness is not guaranteed; a suffixed result is never checked against names
    # produced earlier, so a symbol literally named like an earlier suffixed result (for example
    # a name ending in an underscore and a number) can receive a duplicate capture name.
    # Derives a capture variable name from a symbol name. Input: symbol_name, the original
    # name. Replaces every character that is not a letter, digit or underscore with an underscore,
    # and when that normalized name was already handed out earlier, appends an underscore and the
    # number of previous uses. Returns the unique capture name. Side effect: increments the
    # per-normalized-name usage counter.
    """Translates a symbol name to a unique FileCheck capture name.

    Replaces all characters other than letters, numbers, and underscores with
    underscores. If the resulting name has already been used, appends a counter
    to disambiguate it. For example, this could result in the following sequence
    of replacements:
      1.) "foo.bar.baz" -> "foo_bar_baz"
      2.) "foo.bar_baz" -> "foo_bar_baz_1"
      3.) "foo_bar.baz" -> "foo_bar_baz_2"
      4.) "foo_bar_baz" -> "foo_bar_baz_3"

    Args:
      symbol_name: The original symbol name.

    Returns:
      The generated FileCheck capture name.
    """
    normalized_symbol_name = self._NON_SYMBOL_NAME_CHARS_REGEX.sub(
        "_", symbol_name
    )

    normalized_name_conflict_count = self._num_symbols_with_normalized_name.get(
        normalized_symbol_name, 0
    )
    self._num_symbols_with_normalized_name[normalized_symbol_name] = (
        normalized_name_conflict_count + 1
    )

    optional_disambiguation_suffix = ("" if normalized_name_conflict_count == 0
                                      else f"_{normalized_name_conflict_count}")
    return f"{normalized_symbol_name}{optional_disambiguation_suffix}"


class TestCheckWriter:
  # [CLASS DOC]
  # KNOWN ISSUE: the class name starts with "Test", so test collectors such as pytest may try to
  # collect it as a test class (it has a constructor, so they emit a warning and skip it).
  # Main engine of the tool: transforms HLO test files by running an optimizer on each test case
  # and inserting generated FileCheck directives above it. Holds the optimizer program and
  # arguments, the expansion token, an optional worker count, and a thread pool that exists only
  # while the instance is used as a context manager (with block). Without the context manager,
  # or with a worker count of one, test cases are processed one after another; otherwise they are
  # processed concurrently with results kept in order. Offers splitting and joining of test
  # cases, a per-case transformation helper, the annotation operations, and file printing and
  # in-place overwriting. Constants define the delimiter pattern and delimiter text.
  """Rewrites each test case in an HLO file, optionally overwriting the file."""

  _TEST_CASE_DELIMITER_REGEX: re.Pattern[str] = re.compile(r"^// *-{5,}\n?$")

  _TEST_CASE_DELIMITER_STRING: str = "\n// -----\n\n"

  def __init__(
      self,
      optimizer_path: str,
      optimizer_args: ListOrTuple[str],
      worker_count: Optional[int] = None,
      expand_to_input: str = _DEFAULT_INPUT_FILE_EXPANSION_TOKEN,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: optimizer_path, program to run; optimizer_args, list or tuple of its
    # arguments (copied into a list); worker_count, optional number of worker threads (null means
    # let the pool decide when the context manager is entered, one disables parallelism);
    # expand_to_input, the token replaced by the temporary test case path (default brace
    # placeholder). Raises a value error if worker_count is given and is less than one.
    # Creates no pool yet. Returns nothing.
    """TestCheckWriter constructor.

    Args:
      optimizer_path: The program to use for optimizing the HLO.
      optimizer_args: The arguments to pass into the optimizer tool.
      worker_count: The number of worker threads to use for parallel test-case
        transformations. If `None`, the worker count will be inferred. If 1, or
        if the instance is used without a context manager (i.e. a `with` block),
        the transformations will be performed sequentially.
      expand_to_input: When running the optimizer on a test case, all instances
        of this substring in `optimizer_args` will expand to the path of a
        temporary file containing the text of that test case.
    """
    if worker_count is not None and worker_count < 1:
      raise ValueError(f"The `worker_count` argument must be either `None` or "
                       f"a positive integer; got {worker_count}.")

    self._optimizer_path: str = optimizer_path
    self._optimizer_args: list[str] = list(optimizer_args)
    self._worker_count: Optional[int] = worker_count
    self._expand_to_input: str = expand_to_input

    # The worker pool, if applicable, is created in `__enter__` and destroyed in
    # `__exit__`.
    self._worker_pool: Optional[multiprocessing.pool.Pool] = None
    self._context_manager_active: bool = False

  def __enter__(self) -> TestCheckWriter:
    # [FUNCTION DOC]
    # Context manager entry. Marks the instance as active and, unless the worker count is exactly
    # one, creates a thread pool (with the given worker count, or a default size when it is the
    # null value). Returns the instance itself. Raises a runtime error if the instance is already
    # inside a with block. Side effect: starts worker threads.
    """Context manager setup.

    Initializes `self._worker_pool` if `self._worker_count` is either `None`
    or an integer greater than 1. (In the former case, the worker count will
    be inferred.)

    Returns:
      `self`.

    Raises:
      RuntimeError: If this instance already has an active context manager.
    """
    if self._context_manager_active:
      raise RuntimeError("Tried to enter two simultaneous `with` blocks "
                         "managing the same `TestCheckWriter` instance.")

    self._context_manager_active = True
    assert self._worker_pool is None

    if self._worker_count is None or self._worker_count > 1:
      self._worker_pool = multiprocessing.pool.ThreadPool(self._worker_count)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring and design present this as cleanup, but when the body raised
    # ChildProcessError (the optimizer failed) this writes the message to standard error and
    # returns true, suppressing the exception; the main entry point then returns normally and the
    # process exits with a success status even though no output was produced. Also the pool is
    # closed but never joined or terminated.
    # Context manager exit. Inputs: the exception type, value and traceback (null values when no
    # exception). Closes and drops the worker pool if one exists and marks the instance inactive.
    # Returns true only for ChildProcessError (after printing its text to standard error);
    # returns false for everything else so other exceptions propagate.
    """Context manager cleanup.

    Closes the worker pool if it exists.

    Args:
      exc_type: The type of the raised exception, if any.
      exc_val: The raised exception, if any.
      exc_tb: The traceback of the raised exception, if any.

    Returns:
      `True` if an exception should be suppressed, `False` otherwise.
    """
    if self._worker_pool is not None:
      self._worker_pool.close()
      self._worker_pool = None

    self._context_manager_active = False

    if exc_type is ChildProcessError:
      # In the case of a ChildProcessError, write its message to stderr and then
      # suppress the exception so that its stack trace isn't printed.
      sys.stderr.write(str(exc_val))
      return True

    # Don't suppress other exception types.
    return False

  def split_test_cases(
      self, test_file: Iterator[str]
  ) -> Iterator[Iterator[str]]:
    # [FUNCTION DOC]
    # Generator that splits the lines of a test file into test cases. Input: test_file, iterator
    # over lines. Delimiter lines (double slash, optional spaces, at least five dashes, optional
    # newline) are consumed and not included in any case. Yields one in-memory text stream per test
    # case, rewound to the start, in order; the last one is yielded at the end (so there is always
    # at least one, possibly empty, stream, and a trailing delimiter produces a trailing empty
    # case). Lazy; reads the input while iterating.
    """Splits an HLO test file into test cases delimited by "// -----" lines."""
    output_stream = io.StringIO()
    for line in test_file:
      if self._TEST_CASE_DELIMITER_REGEX.match(line):
        output_stream.seek(0)
        yield output_stream
        output_stream = io.StringIO()
      else:
        output_stream.write(line)
    output_stream.seek(0)
    yield output_stream

  def join_test_cases(
      self,
      test_cases: Union[
          Iterator[Iterator[str]], Iterator[tuple[Iterator[str], ...]]
      ],
      num_outputs: int = 1,
  ) -> Union[Iterator[str], tuple[Iterator[str], ...]]:
    # [FUNCTION DOC]
    # Concatenates per-test-case output streams back into whole-file streams. Inputs: test_cases,
    # iterator over test cases, each either one line iterator (single output) or a tuple of line
    # iterators (multiple outputs); num_outputs, how many streams each case has (default 1, must be
    # at least 1). Returns one line iterator if num_outputs is one, otherwise a tuple of that many
    # iterators. The first output stream gets the delimiter text (blank line, delimiter line,
    # blank line) between cases; additional streams get no delimiter. Raises a value error if
    # num_outputs is less than one. Delegates to the unary or n-ary joiner; the casts are only
    # for type checkers.
    """Concatenates the output stream(s) from each test case in `test_cases`.

    Args:
      test_cases: An iterator over the test cases to join. Each test case is
        either an iterator or a tuple of iterators. Each iterator for test case
        `i` yields a sequence of lines of text corresponding to test case `i`.
        If there are multiple iterators per test case, the first one is the text
        of the test case itself; any additional iterators are secondary outputs
        generated while processing the test case, e.g. "RUN:" directives that
        should be moved to the top of the file.
      num_outputs: The number of output streams to expect from each test case.
        Must be a positive integer.

    Returns:
      `num_outputs` text streams. The `i`th output stream will be the
      concatenation of all `test_cases[:][i]`.

      The first output stream will be treated as the text of the actual test
      cases, so a delimiter of "// -----" will be inserted between test cases.
      Any additional output streams will not have delimiters inserted between
      test cases.

      The `num_outputs` output streams will be returned as a tuple of iterators
      if `num_outputs > 1` or as a single iterator (not wrapped in a tuple) if
      `num_outputs == 1`.
    """
    if num_outputs < 1:
      raise ValueError("The `num_outputs` argument must be a positive integer; "
                       f"got {num_outputs}.")
    elif num_outputs == 1:
      # Delimit test cases with "// -----" (with empty lines before and after).
      delimiter = self._TEST_CASE_DELIMITER_STRING
      # Tell type checkers to narrow the traced type of `test_cases`.
      test_cases = cast(Iterator[Iterator[str]], test_cases)
      # Forward to the implementation handling a single return value.
      return self._join_test_cases_unary(test_cases, delimiter)
    else:
      # In the primary output stream, delimit test cases with "// -----" (with
      # empty lines before and after). Do not insert any test-case delimiter in
      # secondary output streams.
      delimiters = [self._TEST_CASE_DELIMITER_STRING] + [""] * (num_outputs - 1)
      # Tell type checkers to narrow the traced type of `test_cases`.
      test_cases = cast(Iterator[tuple[Iterator[str], ...]], test_cases)
      # Forward to the implementation handling multiple return values.
      return self._join_test_cases_n_ary(test_cases, delimiters)

  def _join_test_cases_unary(
      self,
      test_cases: Iterator[Iterator[str]],
      delimiter: str,
  ) -> Iterator[str]:
    # [FUNCTION DOC]
    # HELPER: single-output joiner for join_test_cases. Inputs: test_cases, iterator over line
    # iterators; delimiter, text yielded between cases. Generator yielding the lines of each case
    # in order, with the delimiter lines before every case except the first. Lazy.
    """Concatenates the streams in `test_cases`, delimited by `delimiter`."""
    # Wrap the delimiter in an `io.StringIO` for line-by-line access.
    delimiter_stream = io.StringIO(delimiter)

    # Don't print the delimiter before the first test case.
    delimiter_stream.seek(0, os.SEEK_END)

    for test_case_stream in test_cases:
      for line in delimiter_stream:
        yield line
      for line in test_case_stream:
        yield line
      delimiter_stream.seek(0)

  def _join_test_cases_n_ary(
      self,
      test_cases: Iterator[tuple[Iterator[str], ...]],
      delimiters: ListOrTuple[str],
  ) -> tuple[Iterator[str], ...]:
    # [FUNCTION DOC]
    # HELPER: multi-output joiner for join_test_cases. Inputs: test_cases, iterator over tuples of
    # N line iterators; delimiters, list or tuple of N delimiter strings, one per output position.
    # Returns a tuple of N rewound in-memory text streams; output i holds the concatenation of the
    # i-th stream of every case, with delimiter i inserted before every case except the first.
    # Unlike the unary joiner this is eager: it consumes all cases (and thus runs all
    # transformations) before returning. Side effect: consumes the test case iterators.
    """Joins each output[i] from all `test_cases`, delimited by `delimiters[i]`.

    Given an iterator to tuples of iterators, this function concatenates along
    the first "dimension" and returns a tuple of concatenated iterators. The
    concatenated iterators comprising `output_stream_tuple[i]` are delimited by
    `delimiters[i]`.

    For example, given the following input:
      `test_cases = [("a", "b"), ("c", "d")]`
      `delimiters = ("-", "_")`
    This function will return a tuple of two iterators:
      `[("a-", "b-"), ("c_", "d_")]`

    Args:
      test_cases: An iterator over the test cases to join. Each test case is a
        tuple of N iterators yielding lines of text.
      delimiters: A list or tuple of N delimiter strings, one for each member of
        the returned tuple `output_stream_tuple`.

    Returns:
      A tuple of N output-stream iterators. For each tuple-element index `i`,
      `output_stream_tuple[i]` is the concatenation of `test_cases[:][i]`,
      delimited by `delimiters[i]`.
    """
    output_stream_tuple, delimiter_stream_tuple = zip(
        *((io.StringIO(), io.StringIO(delimiter)) for delimiter in delimiters),
    )

    # Don't yield the delimiters before the first test case.
    for delimiter_stream in delimiter_stream_tuple:
      delimiter_stream.seek(0, os.SEEK_END)

    for test_case_stream_tuple in test_cases:
      for (
          test_case_stream,
          delimiter_stream,
          output_stream,
      ) in zip(
          test_case_stream_tuple,
          delimiter_stream_tuple,
          output_stream_tuple,
      ):
        output_stream.writelines(delimiter_stream)
        output_stream.writelines(test_case_stream)
        delimiter_stream.seek(0)

    for output_stream in output_stream_tuple:
      output_stream.seek(0)

    return output_stream_tuple

  def for_each_test_case(
      self,
      test_file: Iterator[str],
      transformation: Union[
          Callable[[Iterator[str]], Iterator[str]],
          Callable[[Iterator[str]], tuple[Iterator[str], ...]],
      ],
      num_outputs: int = 1,
  ) -> Union[Iterator[str], tuple[Iterator[str], ...]]:
    # [FUNCTION DOC]
    # Applies a transformation to every test case of a file and joins the results. Inputs:
    # test_file, iterator over file lines; transformation, callback taking one test case's line
    # iterator and returning a line iterator or a tuple of them; num_outputs, number of streams
    # the callback returns (default 1, at least 1). The file is split into cases; the callback is
    # applied to each case, sequentially when there is no worker pool, otherwise concurrently with
    # results returned in input order. Returns what join_test_cases returns for the results (one
    # iterator or a tuple of num_outputs iterators). Raises a value error for an invalid
    # num_outputs; exceptions from the callback surface while the output is consumed.
    """Applies `transformation` to each test case in `test_file`.

    Args:
      test_file: An iterator over the lines of an HLO test file.
      transformation: A function that takes an iterator over the lines of a
        single test case and returns either a single `Iterator[str]` or a tuple
        of `Iterator[str]`s. If it returns a tuple of iterators, the first
        iterator represents the text of the transformed test case; any
        additional iterators represent secondary outputs generated while
        processing the test case, e.g. "RUN:" directives that should be moved to
        the top of the file. Each returned iterator is expected to yield text
        line by line.
      num_outputs: The number of output streams to expect from `transformation`.
        Must be a positive integer.

    Returns:
      `num_outputs` text streams. The `i`th output stream will consist of the
      `i`th result of `transformation(test_case)` concatenated across all test
      cases in `test_file`.

      The first output stream will be treated as the text of the transformed
      test cases, so a delimiter of "// -----" will be inserted between test
      cases. Any additional output streams will not have delimiters inserted
      between test cases.

      The `num_outputs` output streams will be returned as a tuple of iterators
      if `num_outputs > 1` or as a single iterator (not wrapped in a tuple) if
      `num_outputs == 1`.
    """
    test_cases = self.split_test_cases(test_file)

    transformed_test_cases = cast(
        Union[Iterator[Iterator[str]], Iterator[tuple[Iterator[str], ...]]],
        (
            (transformation(test_case) for test_case in test_cases)
            if self._worker_pool is None
            else self._worker_pool.imap(transformation, test_cases)
        ),
    )

    return self.join_test_cases(transformed_test_cases, num_outputs)

  def annotate_test_case(
      self, test_case: Iterator[str]
  ) -> tuple[Iterator[str], Iterator[str]]:
    # [FUNCTION DOC]
    # Transforms one test case: runs it through the optimizer and prepends the generated
    # expectations. Input: test_case, iterator over its lines. Steps: strip check and run
    # directives (recording the run directives), normalize whitespace, duplicate the stream, send
    # one copy through the optimizer runner (this runs the external program immediately and can
    # raise ChildProcessError), and convert its output to check lines. Returns a pair: first an
    # iterator yielding the check lines, one blank line and then the normalized original test
    # case; second an iterator over the stripped run directive lines (to be hoisted to the top
    # of the file). Side effects: temporary file and subprocess via the optimizer runner.
    """Transforms a test case, inserting FileCheck directives above it.

    Args:
      test_case: An iterator over the lines of a single test case.

    Returns:
      A tuple of two iterators. The first iterator yields the annotated test
      case containing FileCheck directives followed by the original HLO IR. The
      second iterator yields any RUN directives that were stripped from the
      original test case.
    """
    split_input = HloStreamSplitter(test_case,
                                    record_directives={DirectiveComment.RUN})

    optimizer_input, passthrough_input = itertools.tee(
        _fix_whitespace(split_input.non_directive_lines()))

    test_case_output = optimize_hlo(optimizer_input,
                                    self._optimizer_path,
                                    self._optimizer_args,
                                    expand_to_input=self._expand_to_input)

    test_case_checks = HloFileCheckLines(test_case_output)

    transformed_test_case = itertools.chain(
        test_case_checks,
        ["\n"],
        passthrough_input,
    )

    run_directives = split_input.directive_history(DirectiveComment.RUN)

    return transformed_test_case, run_directives

  def annotate_test_file(self, test_file: Iterator[str]) -> Iterator[str]:
    # [FUNCTION DOC]
    # Transforms a whole test file. Input: test_file, iterator over its lines. Annotates each
    # test case with per-case two-output processing, then returns an iterator that yields the
    # generated banner comment line, all hoisted run directive lines, a blank line, and then the
    # annotated test cases separated by delimiter lines. Because the two outputs are joined
    # eagerly, all optimizer runs occur before it returns. Can raise ChildProcessError (when an
    # optimizer run fails); uses the worker pool if active.
    """Inserts FileCheck directives above each test case in an HLO test file.

    Args:
      test_file: An iterator over the lines of an HLO test file.

    Returns:
      An iterator over the lines of the transformed HLO test file. Each test
      case is preceded by FileCheck directives describing the expected output
      of the optimizer on that test case.
    """
    transformed_tests, run_directives = self.for_each_test_case(
        test_file,
        self.annotate_test_case,
        num_outputs=2,
    )

    return itertools.chain(
        [_BANNER_COMMENT_LINE],
        run_directives,
        ["\n"],
        transformed_tests,
    )

  def transform_and_print_file(
      self,
      file_path: str,
      transformation: Optional[Callable[[Iterator[str]], Iterator[str]]] = None,
      output_stream: io.TextIOBase = cast(io.TextIOBase, sys.stdout),
  ) -> None:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the default for output_stream is evaluated once, when the class is defined, so
    # it is the standard output object at import time; redirecting standard output later is not
    # honored unless a stream is passed explicitly.
    # Reads a file (or standard input), transforms it and writes the result to a stream.
    # Inputs: file_path, path to read, or the single dash for standard input; transformation,
    # optional callback taking an iterator of lines and returning an iterator of lines (default is
    # annotate_test_file); output_stream, text stream to write to (default standard output).
    # Returns nothing. Side effects: reads the file, writes the transformed lines to the stream;
    # may raise a file error if the path cannot be opened, or whatever the transformation raises.
    """Reads from `file_path`, applies a transformation, and prints to `stdout`.

    Args:
      file_path: The path to the input file. If this is equal to the constant
        `_STANDARD_IO_STREAMS` (i.e. the string "-"), the input will come from
        `stdin`.
      transformation: A function that takes an iterator over the lines of an HLO
        file and returns an iterator over the lines of the transformed file. If
        this is left as `None`, `self.annotate_test_file` will be used.
      output_stream: The stream to which the transformed file should be written.
        Defaults to `stdout`.
    """
    if transformation is None:
      transformation = self.annotate_test_file

    if file_path == _STANDARD_IO_STREAMS:
      # Read from `stdin`, transform the stream, and write to `stdout`.
      output_stream.writelines(transformation(sys.stdin))
    else:
      # Read from `file_path`, transform the stream, and write to `stdout`.
      with open(file_path, "r") as file_contents:
        output_stream.writelines(transformation(file_contents))

  def transform_and_overwrite_file(
      self,
      file_path: str,
      transformation: Optional[Callable[[Iterator[str]], Iterator[str]]] = None,
  ) -> None:
    # [FUNCTION DOC]
    # Transforms a file in place. Inputs: file_path, the file to rewrite; transformation, optional
    # callback from line iterator to line iterator (default annotate_test_file). The transformed
    # text is first written to a new temporary file (not deleted automatically, created in the
    # default temporary directory), and only after the transformation completes is it moved over
    # the original, so a failure during transformation leaves the original untouched (but leaks
    # the temporary file). The move may copy across file systems and the temporary file's
    # permissions replace those of the original. Returns nothing. Can raise file errors or
    # whatever the transformation raises.
    """Transforms the contents of `file_path`, overwriting the file.

    Args:
      file_path: The path to the file whose contents are to be transformed.
      transformation: A function that takes an iterator over the lines of an HLO
        file and returns an iterator over the lines of the transformed file. If
        this is left as `None`, `self.annotate_test_file` will be used.
    """
    if transformation is None:
      transformation = self.annotate_test_file

    # Open the original file for reading and a temporary file for writing.
    with (
        open(file_path, mode="r") as original_file,
        tempfile.NamedTemporaryFile(mode="w", delete=False) as transformed_file,
    ):
      # Read from the original file, transform the stream, and write to the
      # temporary file.
      transformed_file.writelines(transformation(original_file))

    # Move the temporary file, overwriting the original file.
    shutil.move(transformed_file.name, file_path)


def parse_args(
    string_args: Optional[ListOrTuple[str]] = None,
) -> argparse.Namespace:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the help text for the expansion option contains a grammar slip ("will be expand")
  # and the description says "an specified" file; cosmetic only.
  # Parses the command-line arguments of this tool. Input: string_args, optional list or tuple
  # of argument strings (default: the process arguments without the program name). Defines the
  # positional test file (a path or a single dash), the in-place flag (short and long forms,
  # default false), the expansion token option (default brace placeholder), the optimizer command
  # positional and zero or more optimizer arguments. Returns the parsed namespace with attributes
  # test_file, in_place, expand_to_input, opt_cmd and opt_args. If in-place mode is requested while
  # reading standard input it prints a warning to standard error and turns the flag off. The
  # argument parser may print usage and exit the process on invalid input or on a help request.
  """Parses the command-line arguments passed into this script."""
  if string_args is None:
    string_args = sys.argv[1:]

  parser = argparse.ArgumentParser(
      prog=_SCRIPT_NAME,
      usage="%(prog)s TEST_FILE [-h] [-i] [-I EXPAND] -- OPT_CMD [OPT_ARGS...]",
      description=(
          f"For each test case in an specified HLO test file, this script runs "
          f"the test case through an HLO optimizer, converts the optimized HLO "
          f"into FileCheck expectations, and inserts these expectations above "
          f"the test case.\n"
          f"\n"
          f"concrete usage example:\n"
          f"  %(prog)s tests/my_pass.hlo "
          f"-- hlo-opt {_DEFAULT_INPUT_FILE_EXPANSION_TOKEN} --passes=my_pass\n"
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument(
      "test_file",
      metavar="TEST_FILE",
      type=str,
      help=(f'The HLO test file to update, or "{_STANDARD_IO_STREAMS}" for '
            f"stdin/stdout."))
  parser.add_argument(
      "-i", "--in_place",
      action="store_true",
      default=False,
      help=(f"Modify TEST_FILE in place instead of printing the results to "
            f'stdout. (A TEST_FILE argument of "{_STANDARD_IO_STREAMS}" '
            f"overrides this flag.)"))
  parser.add_argument(
      "-I", "--expand_to_input",
      metavar="EXPAND",
      type=str,
      default=_DEFAULT_INPUT_FILE_EXPANSION_TOKEN,
      help=(f"All instances of this substring in OPT_ARGS will be expand to "
            f"the path of a temporary file containing a single test case. "
            f'Defaults to "{_DEFAULT_INPUT_FILE_EXPANSION_TOKEN}".'))

  parser.add_argument(
      "opt_cmd",
      metavar="OPT_CMD",
      type=str,
      help='The HLO optimizer to run, e.g. "hlo-opt".')
  parser.add_argument(
      "opt_args",
      metavar="OPT_ARGS...",
      nargs="*",
      default=[],
      help=(f"Arguments to pass into OPT_CMD. The input file should be "
            f'represented by a literal "{_DEFAULT_INPUT_FILE_EXPANSION_TOKEN}" '
            f"(or the custom token specified with the `-I` flag)."))

  parsed_args = parser.parse_args(string_args)

  if parsed_args.in_place and parsed_args.test_file == _STANDARD_IO_STREAMS:
    sys.stderr.write(f'Warning: Suppressing the "-i"/"--in_place" flag because '
                     f'TEST_FILE is set to "{_STANDARD_IO_STREAMS}" '
                     f"(stdin/stdout).\n\n")
    parsed_args.in_place = False

  return parsed_args


def main() -> None:
  # [FUNCTION DOC]
  # KNOWN ISSUE: when the optimizer fails the writer's context manager swallows the error after
  # printing it, so this function returns normally and the process exits with status zero.
  # Entry point of the tool. Takes no parameters and returns nothing. Parses the command line,
  # creates a writer for the optimizer command and arguments (with the chosen expansion token),
  # enters it as a context manager and either rewrites the file in place or prints the result to
  # standard output. Side effects: file reads and writes, subprocesses, standard output/error.
  args = parse_args()

  with TestCheckWriter(
      args.opt_cmd,
      args.opt_args,
      expand_to_input=args.expand_to_input,
  ) as writer:
    if args.in_place:
      writer.transform_and_overwrite_file(args.test_file)
    else:
      writer.transform_and_print_file(args.test_file)


if __name__ == "__main__":
  main()
