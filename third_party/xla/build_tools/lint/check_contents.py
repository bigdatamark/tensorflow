# [MODULE DOC]
# Command line lint tool that inspects the current git diff and fails if a prohibited regular
# expression appears on an added line. Exposes a result record type (RegexLocation), a path
# filter for diff hunks (filter_hunks_by_path), the matcher (check_diffs) and a main entry point.
# It relies on the sibling diff_parser module to obtain and parse the diff. Uses only the
# standard library so it can run on a vanilla CI runner. No import-time side effects; running as
# a script exits with status 1 when violations are found, otherwise 0.
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
"""Command line tool for checking for regexes in diffs.

Filters `git diff` by path, then checks to make sure no lines matching a regex
have been added in the diff.
"""
import argparse
import dataclasses
import logging  # Intended to run on vanilla Github Actions runner
import re
import sys
from typing import Iterable, Sequence

from build_tools.lint import diff_parser


@dataclasses.dataclass
class RegexLocation:
  # [CLASS DOC]
  # Simple dataclass record describing one prohibited match: the file path, line number of the
  # added line, the full line text and the exact text matched. Instances are created by
  # check_diffs and compared by value (dataclass equality) in tests.
  """Path and line where a prohibited regex was found.

  Attributes:
    path: Path of the file which has the prohibited regex.
    line_number: The number of the offending line.
    line_contents: The text of the offending line.
    matched_text: The exact string matched by the regex.
  """

  path: str
  line_number: int
  line_contents: str
  matched_text: str


def filter_hunks_by_path(
    hunks: Iterable[diff_parser.Hunk],
    *,
    path_regexes: list[str],
    path_regex_exclusions: list[str],
) -> list[diff_parser.Hunk]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the result is a list of file diffs, but it returns a list of
  # hunk objects; it also claims every path is matched by default, which is true only when the
  # inclusion list is empty.
  # Filters diff hunks by file path. Inputs (keyword-only lists): path_regexes, patterns of which
  # at least one must match the hunk's file path (an empty list means match everything), and
  # path_regex_exclusions, patterns of which none may match. Matching is a search anywhere in the
  # path. Returns a new list of the hunks that pass, in original order. No side effects; bad
  # patterns raise a regex compile error.
  """Filters files according to path_regexes.

  If a file matches both a path_regex and a path_regex_exclusion, then
  it will be filtered out.

  Arguments:
    hunks: A sequence of Hunk objects representing the hunks of the diff in the
      change.
    path_regexes: A list of regexes. Paths matching these will pass through the
      filter. By default, every path is matched.
    path_regex_exclusions: A list of regexes. Paths that match both a path_regex
      and a path_regex_exclusion won't pass through the filter.

  Returns:
    A list of FileDiffs whose paths match a path_regex and don't match
      any path_regex_exclusions.
  """

  if not path_regexes:
    path_regexes = [".*"]  # by default match everything

  path_regexes = [re.compile(regex) for regex in path_regexes]

  def should_include(path: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: inclusion test for filter_hunks_by_path.
    # Returns true if any compiled inclusion pattern matches somewhere in the path.
    return any(regex.search(path) for regex in path_regexes)

  path_regex_exclusions = [re.compile(regex) for regex in path_regex_exclusions]

  def should_exclude(path: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: exclusion test for filter_hunks_by_path.
    # Returns true if any compiled exclusion pattern matches somewhere in the path.
    return any(regex.search(path) for regex in path_regex_exclusions)

  return [
      hunk
      for hunk in hunks
      if should_include(hunk.file) and not should_exclude(hunk.file)
  ]


def check_diffs(
    hunks: Iterable[diff_parser.Hunk],
    *,
    prohibited_regex: str,
    suppression_regex: str | None = None,
) -> list[RegexLocation]:
  # [FUNCTION DOC]
  # Scans added lines of diff hunks for a prohibited pattern. Inputs: hunks (iterable of parsed
  # hunk objects), prohibited_regex (pattern string) and optional suppression_regex (pattern
  # string or nothing). For each added line (with its line number) not containing the suppression
  # pattern, every non-overlapping match of the prohibited pattern yields a location record.
  # Returns the list of RegexLocation in encounter order (empty when clean). Raises a regex
  # compile error for invalid patterns. No other side effects.
  """Checks FileDiffs for prohibited regexes.

  Arguments:
    hunks: A sequence of Hunk objects representing the hunks of the diff.
    prohibited_regex: The regex that isn't allowed in the diff.
    suppression_regex: A regex used as an escape hatch to allow the prohibited
      regex in the diff. If this is found on the same line as prohibited_regex,
      there is no error.

  Returns:
    A list of RegexLocations where the prohibited_regex is found.
  """

  prohibited_regex = re.compile(prohibited_regex)
  if suppression_regex is not None:
    suppression_regex = re.compile(suppression_regex)

  def should_not_suppress(line) -> bool:
    # [FUNCTION DOC]
    # HELPER: suppression test for check_diffs.
    # Returns true if there is no suppression pattern or it does not occur in the given line text.
    if suppression_regex:
      return not suppression_regex.search(line)
    return True

  regex_locations = []
  for hunk in hunks:
    for line_no, line in hunk.added_lines():
      if should_not_suppress(line):
        regex_locations.extend(
            [
                RegexLocation(hunk.file, line_no, line, regex_match.group())
                for regex_match in prohibited_regex.finditer(line)
            ]
        )

  return regex_locations


def main(argv: Sequence[str]):
  # [FUNCTION DOC]
  # Command line entry point. Input: argv, the full argument list whose first element (program
  # name) is ignored. Flags: optional lists of path patterns and exclusion patterns, a required
  # prohibited pattern, an optional suppression pattern and a required failure message. Gets the
  # git diff output, parses it into hunks, filters by path and checks for prohibited matches. On
  # matches, logs an error per match (location, matched text and line) plus the failure message
  # and exits the process with status 1; otherwise logs an info message and exits with status 0.
  # Never returns normally. Invalid arguments make argparse exit with an error.
  parser = argparse.ArgumentParser(
      description="Check `git diff` for prohibited regexes."
  )
  parser.add_argument("--path_regex", nargs="*", default=[])
  parser.add_argument("--path_regex_exclusion", nargs="*", default=[])
  parser.add_argument("--prohibited_regex", required=True)
  parser.add_argument("--suppression_regex")
  parser.add_argument("--failure_message", required=True)

  # We don't want to include path/to/check_contents.py as an argument
  args = parser.parse_args(argv[1:])

  file_diffs = filter_hunks_by_path(
      diff_parser.parse_hunks(diff_parser.get_git_diff_stdout()),
      path_regexes=args.path_regex,
      path_regex_exclusions=args.path_regex_exclusion,
  )

  regex_locations = check_diffs(
      file_diffs,
      prohibited_regex=args.prohibited_regex,
      suppression_regex=args.suppression_regex,
  )

  if regex_locations:
    for loc in regex_locations:
      logging.error(
          "Found `%s` in %s:%s",
          args.prohibited_regex,
          loc.path,
          loc.line_number,
      )
      logging.error(
          "Matched `%s` in line `%s`", loc.matched_text, loc.line_contents
      )
      logging.error("Failure message: %s", args.failure_message)
    sys.exit(1)
  else:
    logging.info(
        "Prohibited regex `%s` not found in diff!", args.prohibited_regex
    )
    sys.exit(0)


if __name__ == "__main__":
  main(sys.argv)
