# [MODULE DOC]
# Unit tests for the check_contents lint module: path filtering of diff hunks and detection of a
# prohibited pattern (with and without a suppression pattern) using golden diff files from the
# testdata directory. Runs under the absl test runner.
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
from absl.testing import absltest

from build_tools import test_utils
from build_tools.lint import check_contents
from build_tools.lint import diff_parser


class CheckDiffsTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for filter_hunks_by_path and check_diffs. Shared fixtures: parsed hunks of a diff with
  # a bad source file and an important source file, loaded once for the class.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture: reads two diff files (one adding a prohibited name in a source file
    # under a directory, one for an important file) from the lint testdata directory and stores
    # the parsed hunks as class attributes. Raises a file error if missing.
    super().setUpClass()

    testdata = test_utils.xla_src_root() / "build_tools" / "lint" / "testdata"
    with (testdata / "bad_cc.diff").open() as f:
      cls.bad_cc_hunks = diff_parser.parse_hunks(f.read())

    with (testdata / "important_cc.diff").open() as f:
      cls.important_cc_hunks = diff_parser.parse_hunks(f.read())

  def test_check_good_diff(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the empty-check call passes an extra argument (zero) as the failure message
    # parameter, which is harmless but unintended.
    # Scenario: check the bad diff with a prohibited name and a suppression pattern that is present
    # on the offending lines. Asserts no violations are reported.

    locs = check_contents.check_diffs(
        self.bad_cc_hunks,
        prohibited_regex="Make_Unique",
        suppression_regex="OK",
    )
    self.assertEmpty(locs, 0)

  def test_check_suppressed_diff_without_suppressions(self):
    # [FUNCTION DOC]
    # Scenario: check the bad diff for the prohibited name without a suppression pattern. Asserts
    # exactly two violation records, for lines 3 and 6 of the bad source file, with the expected
    # line contents and matched text.

    locs = check_contents.check_diffs(
        self.bad_cc_hunks, prohibited_regex="Make_Unique"
    )

    expected_locs = [
        check_contents.RegexLocation(
            path="src/dir/bad.cc",
            line_number=3,
            line_contents="using Make_Unique = std::make_unique; // OK",
            matched_text="Make_Unique",
        ),
        check_contents.RegexLocation(
            path="src/dir/bad.cc",
            line_number=6,
            line_contents="  return Make_Unique<int>(a + b); // OK. Fixed now!",
            matched_text="Make_Unique",
        ),
    ]

    self.assertEqual(locs, expected_locs)

  def test_check_suppressed_diff_with_path_regexes(self):
    # [FUNCTION DOC]
    # Scenario: filter hunks to those whose path matches the important-file pattern. Asserts one
    # hunk remains and that checking it for the prohibited name finds nothing.
    filtered_hunks = check_contents.filter_hunks_by_path(
        self.bad_cc_hunks,
        path_regexes=["src/important\\..*"],
        path_regex_exclusions=[],
    )

    self.assertLen(filtered_hunks, 1)

    locs = check_contents.check_diffs(
        filtered_hunks, prohibited_regex="Make_Unique"
    )

    self.assertEmpty(locs)

  def test_check_suppressed_diff_with_exclusions(self):
    # [FUNCTION DOC]
    # Scenario: filter with an exclusion pattern for the directory of the bad file and no
    # inclusion patterns. Asserts one hunk remains and checking it finds no violations.
    filtered_hunks = check_contents.filter_hunks_by_path(
        self.bad_cc_hunks,
        path_regexes=[],
        path_regex_exclusions=["src/dir/.*"],
    )

    self.assertLen(filtered_hunks, 1)

    locs = check_contents.check_diffs(
        filtered_hunks, prohibited_regex="Make_Unique"
    )

    self.assertEmpty(locs)

  def test_check_suppressed_diff_with_suppression(self):
    # [FUNCTION DOC]
    # Scenario: filtering with empty inclusion and exclusion lists. Asserts the result equals the
    # original hunks (no-op), then that checking with the suppression pattern reports nothing.

    filtered_hunks = check_contents.filter_hunks_by_path(
        self.bad_cc_hunks, path_regexes=[], path_regex_exclusions=[]
    )

    # filtering without path_regex(_exclusions) is a noop
    self.assertEqual(self.bad_cc_hunks, filtered_hunks)

    locs = check_contents.check_diffs(
        filtered_hunks, prohibited_regex="Make_Unique", suppression_regex="OK"
    )

    self.assertEmpty(locs)


if __name__ == "__main__":
  absltest.main()
