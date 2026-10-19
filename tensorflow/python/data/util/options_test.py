# [MODULE DOC]
# Unit tests for the tf.data options utilities module: option property creation and type
# checking, documentation, merging (flat, nested, error cases), immutability and unknown
# attribute handling. Defines two small fixture options classes. Runs tests when executed.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dataset options utilities."""

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import options
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class _TestOptions(options.OptionsBase):
  # [CLASS DOC]
  # HELPER: test fixture options class with an integer option (default forty-two) and a float
  # option (default about 3.14), each with a short documentation string.
  x = options.create_option(
      name="x",
      ty=int,
      docstring="the answer to everything",
      default_factory=lambda: 42)
  y = options.create_option(
      name="y", ty=float, docstring="a tasty pie", default_factory=lambda: 3.14)


class _NestedTestOptions(options.OptionsBase):
  # [CLASS DOC]
  # HELPER: test fixture options class with one option that must hold a test options object
  # (default nothing/null). Used to test recursive merging and incompatible-type merging.
  opts = options.create_option(
      name="opts", ty=_TestOptions, docstring="nested options")


class OptionsTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test class for the options utilities (option property creation, merging, immutability and
  # attribute checks). Uses the two small options classes defined above as fixtures.

  @combinations.generate(test_base.default_test_combinations())
  def testDocumentation(self):
    # [FUNCTION DOC]
    # Test: the documentation text given when creating an option is exposed as the documentation
    # of the resulting property, for both test options.
    self.assertEqual(_TestOptions.x.__doc__, "the answer to everything")
    self.assertEqual(_TestOptions.y.__doc__, "a tasty pie")

  @combinations.generate(test_base.default_test_combinations())
  def testCreateOption(self):
    # [FUNCTION DOC]
    # Test: created option properties. Asserts defaults and types, that assignment of a value of
    # the right type works, and that a wrong type (float for the integer option, integer for the
    # float option) raises a type error.
    opts = _TestOptions()
    self.assertEqual(opts.x, 42)
    self.assertEqual(opts.y, 3.14)
    self.assertIsInstance(opts.x, int)
    self.assertIsInstance(opts.y, float)
    opts.x = 0
    self.assertEqual(opts.x, 0)
    with self.assertRaises(TypeError):
      opts.x = 3.14
    opts.y = 0.0
    self.assertEqual(opts.y, 0.0)
    with self.assertRaises(TypeError):
      opts.y = 42

  @combinations.generate(test_base.default_test_combinations())
  def testMergeOptions(self):
    # [FUNCTION DOC]
    # Test: merging flat options. Merging with no arguments raises a value error; merging two
    # defaults yields defaults; after setting one field on each object the merged result takes
    # each changed value.
    options1, options2 = _TestOptions(), _TestOptions()
    with self.assertRaises(ValueError):
      options.merge_options()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.x, 42)
    self.assertEqual(merged_options.y, 3.14)
    options1.x = 0
    options2.y = 0.0
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.x, 0)
    self.assertEqual(merged_options.y, 0.0)

  @combinations.generate(test_base.default_test_combinations())
  def testMergeNestedOptions(self):
    # [FUNCTION DOC]
    # Test: merging options holding a nested options object. Asserts the nested option stays unset
    # when neither has it, is taken from whichever side sets it, and when both set it the nested
    # fields are merged recursively (one field from each side).
    options1, options2 = _NestedTestOptions(), _NestedTestOptions()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts, None)
    options1.opts = _TestOptions()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts, _TestOptions())
    options2.opts = _TestOptions()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts, _TestOptions())
    options1.opts.x = 0
    options2.opts.y = 0.0
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts.x, 0)
    self.assertEqual(merged_options.opts.y, 0.0)

  @combinations.generate(test_base.default_test_combinations())
  def testImmutable(self):
    # [FUNCTION DOC]
    # Test: after marking an options object immutable, assigning any attribute raises a value
    # error whose message says mutating the returned options has no effect.
    test_options = _TestOptions()
    test_options._set_mutable(False)

    with self.assertRaisesRegex(
        ValueError, r"Mutating `tf.data.Options\(\)` returned by "
        r"`tf.data.Dataset.options\(\)` has no effect. Use "
        r"`tf.data.Dataset.with_options\(options\)` to set or "
        "update dataset options."):
      test_options.test = 100

  @combinations.generate(test_base.default_test_combinations())
  def testNoSpuriousAttrs(self):
    # [FUNCTION DOC]
    # Test: assigning an unknown attribute raises an attribute error naming the property and
    # class, and reading an unknown attribute also raises an attribute error.
    test_options = _TestOptions()

    with self.assertRaisesRegex(
        AttributeError, "Cannot set the property wrong_attr on _TestOptions."):
      test_options.wrong_attr = True
    with self.assertRaises(AttributeError):
      _ = test_options.wrong_attr

  @combinations.generate(test_base.default_test_combinations())
  def testMergeNoOptions(self):
    # [FUNCTION DOC]
    # Test: merging with no arguments raises a value error saying at least one options object
    # must be provided.
    with self.assertRaisesRegex(ValueError,
                                "At least one options should be provided"):
      options.merge_options()

  @combinations.generate(test_base.default_test_combinations())
  def testMergeOptionsDifferentType(self):
    # [FUNCTION DOC]
    # Test: merging options objects of two different classes raises a type error about
    # incompatible option types, naming both classes.
    options1, options2 = _TestOptions(), _NestedTestOptions()
    with self.assertRaisesRegex(
        TypeError, r"Could not merge incompatible options of type "
        r"\<class \'__main__._NestedTestOptions\'\> and "
        r"\<class \'__main__._TestOptions\'\>."):
      options.merge_options(options1, options2)

  @combinations.generate(test_base.default_test_combinations())
  def testMergeOptionsWrongType(self):
    # [FUNCTION DOC]
    # Test: merging plain integers (not options objects) raises a type error saying all options
    # must inherit from the options base class.
    with self.assertRaisesRegex(
        TypeError, "All options to be merged should inherit from "
        r"\`OptionsBase\` but found option of type \<class \'int\'\> which "
        "does not."):
      options.merge_options(1, 2, 3)


if __name__ == "__main__":
  test.main()
