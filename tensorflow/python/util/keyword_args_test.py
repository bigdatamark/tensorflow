# [MODULE DOC]
# Test module for the keyword-arguments-only decorator, checking accepted keyword calls and
# rejected positional or mixed calls.
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
"""Keyword args tests."""

from tensorflow.python.platform import test
from tensorflow.python.util import keyword_args


class KeywordArgsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the keyword_args_only decorator.

  def test_keyword_args_only(self):
    # [FUNCTION DOC]
    # Scenario: a plain two-argument function and a decorated function that delegates to it.
    # Asserts: the plain function works positionally and by keyword; the decorated one works by
    # keyword; calling the decorated one with all positional or with mixed positional and
    # keyword arguments raises a value error saying it only accepts keyword arguments.

    def func_without_decorator(a, b):
      # [FUNCTION DOC]
      # HELPER: returns the sum of its two arguments.
      return a + b

    @keyword_args.keyword_args_only
    def func_with_decorator(a, b):
      # [FUNCTION DOC]
      # HELPER: decorated function; delegates to the undecorated function and returns the sum.
      return func_without_decorator(a, b)

    self.assertEqual(3, func_without_decorator(1, 2))
    self.assertEqual(3, func_without_decorator(a=1, b=2))
    self.assertEqual(3, func_with_decorator(a=1, b=2))

    # Providing non-keyword args should fail.
    with self.assertRaisesRegex(
        ValueError, "only accepts keyword arguments"):
      self.assertEqual(3, func_with_decorator(1, 2))

    # Partially providing keyword args should fail.
    with self.assertRaisesRegex(
        ValueError, "only accepts keyword arguments"):
      self.assertEqual(3, func_with_decorator(1, b=2))


if __name__ == "__main__":
  test.main()
