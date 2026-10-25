# [MODULE DOC]
# Helper module for the lint tools that turns unified git diff text into structured hunk records.
# Exposes the Hunk record (file, start line, length, raw lines) with an added-lines iterator,
# a generic batching utility, a diff parser (parse_hunks) and a function that runs git diff
# against the main branch (get_git_diff_stdout). Used by the check_contents lint tool.
# Import-time side effects: defines a type variable only.
# HELPER: support module for the lint command line tools.
# Copyright 2023 The OpenXLA Authors.
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
# ============================================================================
"""Parses diffs into hunks."""
import dataclasses
import re
import subprocess
from typing import Generator, Iterable, TypeVar

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class Hunk:
  # [CLASS DOC]
  # Immutable (frozen dataclass) record of one diff hunk: file is the path of the changed file in
  # the new version, start is the first line number in the new file, length is the number of new
  # file lines the header reports, and lines is the list of raw hunk lines (each prefixed with a
  # plus, minus or space marker). Created by parse_hunks; consumed by check_contents.
  """Represents a hunk of a diff."""

  file: str
  start: int
  length: int
  lines: list[str]

  def added_lines(self) -> Generator[tuple[int, str], None, None]:
    # [FUNCTION DOC]
    # Generator over the lines added by this hunk. Takes no inputs. Walks the raw lines tracking
    # the line number in the new file, starting at the hunk start: a line beginning with a plus
    # yields a pair of its number and its text without the plus, then advances the counter; lines
    # beginning with a minus are skipped without advancing; all other (context) lines advance the
    # counter. No side effects.
    current_line_no = self.start
    for line in self.lines:
      if line.startswith("+"):
        yield current_line_no, line[1:]  # elide leading '+'
        current_line_no += 1
      elif line.startswith("-"):
        continue
      else:
        current_line_no += 1


def batch(
    iterable: Iterable[_T], n: int
) -> Generator[tuple[_T, ...], None, None]:
  # [FUNCTION DOC]
  # HELPER: chunking utility for parse_hunks (stand-in for a standard library batching function).
  # KNOWN ISSUE: a final partial chunk (fewer than n items) is silently dropped, because the
  # failure of the inner list construction ends the generator before yielding it.
  # Generator splitting an iterable into tuples of exactly n items. Inputs: iterable and n (chunk
  # size, positive integer). Yields tuples in order; stops when the iterable runs out. Consumes
  # the iterable. No exceptions are expected for n of at least one.
  """Splits an iterable into chunks of size n.

  TODO(ddunleavy): once python 3.12 is available, use itertools.batch.

  Arguments:
    iterable: the iterable to batch.
    n: the number of elements in each batch.

  Yields:
    A tuple of length n of the type that the iterable produces.
  """
  iterator = iter(iterable)
  while True:
    try:
      # Unnecessary list here, but a generator won't raise StopIteration,
      # instead it will raise RuntimeError: "generator raises StopIteration".
      # I'd rather have a list comprehension in place of a generator expression
      # than catch RuntimeError and have to inspect the payload to verify it's
      # the one I want to be catching.
      yield tuple([next(iterator) for _ in range(n)])
    except StopIteration:
      return


def parse_hunks(diff: str) -> list[Hunk]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the hunk split call passes the multiline flag as the third positional argument,
  # which is the maximum number of splits, not flags; so at most 8 splits happen per file
  # (limiting how many hunks per file are parsed) and the flag has no regex effect.
  # Parses unified git diff text into hunk records. Input: diff, the raw git diff output string.
  # Splits the text by per-file header patterns (capturing the new file name, tolerating an
  # optional file mode line), then splits each file's text by hunk headers capturing the new
  # start line and length. Each hunk body is split on newlines (dropping one trailing empty
  # element) to form the lines. Returns a list of Hunk records in order (empty if no diff found).
  # Pure function; raises a value error if captured numbers fail integer conversion.
  """Parses a diff into hunks.

  Arguments:
    diff: The raw output of git diff.

  Returns:
    A list of Hunks.
  """
  diff_pattern = (
      r"diff --git a/.* b/(.*)\n"  # capture file name
      r"(?:\w+ file mode \d+\n)?"  # maybe 'new file mode 100644' or similar
      r"index .*\n"
      r"--- .*\n"
      r"\+\+\+ .*\n"
  )

  # capture line number and length from header
  hunk_header_pattern = r"@@ -\d+,\d+ \+(\d+),(\d+) @@.*\n"

  # ignore initial empty match
  raw_per_file_hunks = re.split(diff_pattern, diff)[1:]

  parsed_hunks = []

  for file, raw_hunks in batch(raw_per_file_hunks, 2):
    # ignore initial empty match
    hunks = re.split(hunk_header_pattern, raw_hunks, re.MULTILINE)[1:]
    for start, length, body in batch(hunks, 3):
      lines = body.split("\n")
      lines = lines if lines[-1] else lines[:-1]  # trim empty line
      parsed_hunks.append(Hunk(file, int(start), int(length), lines))

  return parsed_hunks


def get_git_diff_stdout() -> str:
  # [FUNCTION DOC]
  # Runs the version control diff command comparing the origin main branch to HEAD, in the current
  # working directory, and returns its standard output as text. No inputs. Raises a
  # called-process error if the command fails (for example no such remote branch or not a
  # repository). Side effect: spawns a subprocess.
  """Run git diff with appropriate arguments and capture stdout as a str."""
  proc = subprocess.run(
      ["git", "diff", "origin/main", "HEAD"],
      capture_output=True,
      check=True,
      text=True,
  )
  return proc.stdout
