# [MODULE DOC]
# Unit tests for two small private helpers of the saved_model_cli module: the routine that
# trims and filters lists of strings and the routine that parses a comma-separated op denylist
# into a set. Uses the TensorFlow test case base class combined with parameterized tests.
# No import-time side effects other than imports; runs the test main when executed directly.
# tensorflow/python/tools/saved_model_cli_sanitize_list_test.py
# Copyright 2025 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0
# pylint: disable=line-too-long

"""Unit tests for trim & validation helpers in saved_model_cli."""

from absl.testing import parameterized
from tensorflow.python.platform import test
from tensorflow.python.tools import saved_model_cli as smcli


class SanitizeNonEmptyStrListTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the list-sanitizing helper of the saved model command-line tool: that it
  # trims, drops empty and null items, and complains when nothing is left. No shared fixtures.

  def test_trims_and_drops_empty_and_none(self):
    # [FUNCTION DOC]
    # Scenario: a list mixing padded strings, an empty string, a whitespace-only string and a
    # null item is passed with a field name. Asserts the result is the two real items with
    # whitespace removed, in the original order.
    self.assertEqual(
        smcli._sanitize_nonempty_str_list(
            [' a ', '', '\t', None, 'b '], 'field'
        ),
        ['a', 'b'],
    )

  def test_raises_on_all_empty_like_inputs(self):
    # [FUNCTION DOC]
    # Scenario: every item is blank, whitespace-only, a newline or null. Asserts a value error
    # is raised whose message mentions the field name and the need for at least one non-empty
    # item.
    with self.assertRaisesRegex(ValueError, 'field.*at least one non-empty'):
      smcli._sanitize_nonempty_str_list(['  ', '\n', '', None], 'field')


class DenylistParsingTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case covering parsing of the op denylist flag string into a set of op
  # names by the saved model command-line tool. No shared fixtures.

  @parameterized.parameters(
      ('OpA, OpB, , ,,OpC', {'OpA', 'OpB', 'OpC'}),
      ('  , , ', set()),
      ('', set()),
      (' ReadFile ,WriteFile , PrintV2', {'ReadFile', 'WriteFile', 'PrintV2'}),
  )
  def test_get_op_denylist_set_trims_and_ignores_empties(self, raw, expected):
    # [FUNCTION DOC]
    # Parameterized scenario: each case supplies a raw comma-separated string (with stray
    # spaces, empty items, only separators, or the empty string) and the expected set of op
    # names. Asserts the parser returns exactly that set, so blanks vanish and an empty or
    # separator-only string yields an empty set.
    self.assertEqual(smcli._get_op_denylist_set(raw), expected)


if __name__ == '__main__':
  test.main()
