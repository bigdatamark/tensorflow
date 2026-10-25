# [MODULE DOC]
# Tests for the generated raw ops interface: raw ops accept keyword arguments only, give helpful
# errors, honor names and defaults, share docs with their snake-case aliases, and report
# kernel-level errors for invalid inputs. Runs in both graph and eager modes; when executed as a
# script it enables eager execution first.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Raw ops tests."""

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class RawOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the raw ops wrappers (math, string and data-flow ops) in both graph and
  # eager modes; it is parameterized for the n-gram split cases and has no shared fixtures.

  def testSimple(self):
    # [FUNCTION DOC]
    # Calls the raw add op with keyword arguments on a constant one and checks the evaluated
    # result equals the list containing two.
    x = constant_op.constant(1)
    self.assertEqual([2], self.evaluate(gen_math_ops.Add(x=x, y=x)))

  def testRequiresKwargs(self):
    # [FUNCTION DOC]
    # Calls the raw add op with positional arguments and asserts a type error whose message says
    # that only keyword arguments are accepted.
    with self.assertRaisesRegex(TypeError, "only takes keyword args"):
      gen_math_ops.Add(1., 1.)

  def testRequiresKwargs_providesSuggestion(self):
    # [FUNCTION DOC]
    # Calls the raw add op with one positional and one keyword argument and asserts the type
    # error message lists the possible keyword names (x, y and name).
    msg = "possible keys: \\['x', 'y', 'name'\\]"
    with self.assertRaisesRegex(TypeError, msg):
      gen_math_ops.Add(1., y=2.)

  def testName(self):
    # [FUNCTION DOC]
    # Creates a raw add op with an explicit name; in graph mode only, asserts the result's name
    # is the given name with an output index suffix (names are unavailable in eager mode).
    x = constant_op.constant(1)
    op = gen_math_ops.Add(x=x, y=x, name="double")
    if not context.executing_eagerly():
      # `Tensor.name` is not available in eager.
      self.assertEqual(op.name, "double:0")

  def testDoc(self):
    # [FUNCTION DOC]
    # Asserts the capitalized raw add op and the snake-case add function share the same
    # documentation string.
    self.assertEqual(gen_math_ops.add.__doc__, gen_math_ops.Add.__doc__)

  def testDefaults(self):
    # [FUNCTION DOC]
    # Calls the raw any-reduction op on a one-by-one boolean tensor with and without explicitly
    # passing the keep-dims attribute as false and asserts both results are close, showing the
    # default is false.
    x = constant_op.constant([[True]])
    self.assertAllClose(
        gen_math_ops.Any(input=x, axis=0),
        gen_math_ops.Any(input=x, axis=0, keep_dims=False))

  @parameterized.parameters([[0, 8]], [[-1, 6]])
  def testStringNGramsBadDataSplits(self, splits):
    # [FUNCTION DOC]
    # Parameterized over two invalid split lists (one ending past the data, one with a negative
    # first value). Runs the string n-grams op on six strings with width two and no padding and
    # asserts an invalid-argument error mentioning an invalid split value or that the first
    # split must be zero.
    data = ["aa", "bb", "cc", "dd", "ee", "ff"]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Invalid split value|First split value must be 0"):
      self.evaluate(
          gen_string_ops.string_n_grams(
              data=data,
              data_splits=splits,
              separator="",
              ngram_widths=[2],
              left_pad="",
              right_pad="",
              pad_width=0,
              preserve_short_sequences=False))

  def testStringSplit(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says string split but the test exercises the string n-grams op.
    # Checks two invalid-argument failures of the n-grams op: a negative pad width with
    # preserved short sequences must report that pad width should be non-negative, and
    # huge n-gram widths with a large negative pad width must report a possible integer
    # overflow.
    data = ["123456"]
    data_splits = [0, 1]
    separator = "a" * 15
    ngram_widths = []
    pad_width = -5
    left_pad = right_pad = ""
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Pad width should be >= 0"):
      self.evaluate(gen_string_ops.string_n_grams(
          data=data,
          data_splits=data_splits,
          separator=separator,
          ngram_widths=ngram_widths,
          left_pad=left_pad,
          right_pad=right_pad,
          pad_width=pad_width,
          preserve_short_sequences=True))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Pad width could lead to integer overflow"):
      self.evaluate(
          gen_string_ops.string_n_grams(
              data=["000.0", "000.0"],
              data_splits=[0, 2],
              separator="",
              ngram_widths=[2**30, 2**30],
              left_pad=" ",
              right_pad=" ",
              pad_width=-2**30,
              preserve_short_sequences=False))

  def testGetSessionHandle(self):
    # [FUNCTION DOC]
    # Only in eager mode: calls the get-session-handle raw op and asserts a failed-precondition
    # error saying it was called on null session state. Does nothing in graph mode.
    if context.executing_eagerly():
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "GetSessionHandle called on null session state"):
        gen_data_flow_ops.GetSessionHandle(value=[1])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
