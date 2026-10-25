# [MODULE DOC]
# KNOWN ISSUE: the module docstring says it tests the logging kernels, but the file actually tests
# the string formatting operation (string_ops.string_format) that substitutes tensor summaries into
# a template string.
# Tests for string_format in graph and eager modes: single and multiple tensors and variables,
# summarisation of long dimensions with different summarize settings, nested output layout
# for one to three dimensions, template prefix and suffix, custom placeholder text, mismatch
# between template placeholders and supplied tensors (value error), and unicode text.
# Run via the standard test main entry point. No import-time side effects. Not a helper module.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.kernels.logging_ops."""


from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class StringFormatOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the string formatting operation. Every test runs in both graph and eager mode,
  # builds tensors or variables inside a cached session, formats them into a template and compares
  # the decoded text with the expected summarised rendering. Default summarisation shows
  # three items at the start and end of each dimension.

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDim(self):
    # [FUNCTION DOC]
    # Formats a range of ten integers with a bare placeholder, once passing the tensor directly and
    # once inside a list, and asserts both give the elided text showing the first three and last
    # three values. Returns nothing.
    with self.cached_session():
      tensor = math_ops.range(10)
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(10)
      format_output = string_ops.string_format("{}", [tensor])
      out = self.evaluate(format_output)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneVariableScalar(self):
    # [FUNCTION DOC]
    # Formats a scalar float variable; in graph mode first runs the variable initialiser.
    # Asserts the text is the plain number. Returns nothing.
    with self.cached_session():
      var = variables.Variable(3.34)
      format_output = string_ops.string_format("{}", [var])
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
      out = self.evaluate(format_output)
      expected = "3.34"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneVariableOneDim(self):
    # [FUNCTION DOC]
    # Formats a variable initialised from a range of ten integers (initialising it first in graph
    # mode) and asserts the elided vector text. Returns nothing.
    with self.cached_session():
      var = variables.Variable(math_ops.range(10))
      format_output = string_ops.string_format("{}", [var])
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
      out = self.evaluate(format_output)
      expected = "[0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatTwoVariablesWithAssignAdd(self):
    # [FUNCTION DOC]
    # Creates a scalar float variable and a vector variable, runs an add-one update on the scalar,
    # then formats both with a two-placeholder template. Asserts the text shows the updated scalar
    # followed by the elided vector, proving formatting reads current variable values.
    # Returns nothing.
    with self.cached_session():
      var_one = variables.Variable(2.14)
      plus_one = var_one.assign_add(1.0)
      var_two = variables.Variable(math_ops.range(10))
      format_output = string_ops.string_format("{}, {}", [var_one, var_two])
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
      self.evaluate(plus_one)
      out = self.evaluate(format_output)
      expected = "3.14, [0 1 2 ... 7 8 9]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimFloat(self):
    # [FUNCTION DOC]
    # Formats a vector of eight floats and asserts the elided text, with whole numbers printed
    # without a decimal part. Returns nothing.
    with self.cached_session():
      tensor = constant_op.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = "[0 0.1 0.2 ... 0.5 0.6 0.7]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimMatchesSummarize(self):
    # [FUNCTION DOC]
    # Formats six integers with summarize equal to three and asserts nothing is elided because the
    # length equals twice the summarize value. Returns nothing.
    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=3)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4 5]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimVarySummarize(self):
    # [FUNCTION DOC]
    # Formats six integers with summarize values of minus one (show everything), one, two and ten,
    # asserting full output, first and last one, first and last two, and full output again.
    # Returns nothing.
    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=-1)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4 5]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=1)
      out = self.evaluate(format_output)
      expected = "[0 ... 5]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=2)
      out = self.evaluate(format_output)
      expected = "[0 1 ... 4 5]"
      self.assertEqual(compat.as_text(out), expected)

    with self.cached_session():
      tensor = math_ops.range(6)
      format_output = string_ops.string_format("{}", tensor, summarize=10)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4 5]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorOneDimAlmostSummarize(self):
    # [FUNCTION DOC]
    # Formats five integers with summarize three and asserts nothing is elided since five is
    # below twice the summarize value. Returns nothing.
    with self.cached_session():
      tensor = math_ops.range(5)
      format_output = string_ops.string_format("{}", tensor, summarize=3)
      out = self.evaluate(format_output)
      expected = "[0 1 2 3 4]"
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTwoDimLessThanSummarize(self):
    # [FUNCTION DOC]
    # Formats a two-by-two matrix with summarize three and asserts the full two-line matrix text
    # with no elision. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(4), [2, 2])
      format_output = string_ops.string_format("{}", tensor, summarize=3)
      out = self.evaluate(format_output)
      expected = ("[[0 1]\n"
                  " [2 3]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTwoDim(self):
    # [FUNCTION DOC]
    # Formats a ten-by-ten matrix with default summarisation and asserts the multi-line text
    # showing the first three and last three rows with elision lines and elided columns.
    # Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = ("[[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTwoDimSummarizeTwo(self):
    # [FUNCTION DOC]
    # Formats a ten-by-ten matrix with summarize two and asserts only the first two and last two
    # rows and columns are shown. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("{}", tensor, summarize=2)
      out = self.evaluate(format_output)
      expected = ("[[0 1 ... 8 9]\n"
                  " [10 11 ... 18 19]\n"
                  " ...\n"
                  " [80 81 ... 88 89]\n"
                  " [90 91 ... 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorThreeDim(self):
    # [FUNCTION DOC]
    # Formats a ten-by-ten-by-ten tensor with default summarisation and asserts the full nested
    # text: blocks separated by blank lines, three leading and trailing blocks, an elision line
    # between them, and the same elision inside each block. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(1000), [10, 10, 10])
      format_output = string_ops.string_format("{}", tensor)
      out = self.evaluate(format_output)
      expected = ("[[[0 1 2 ... 7 8 9]\n"
                  "  [10 11 12 ... 17 18 19]\n"
                  "  [20 21 22 ... 27 28 29]\n"
                  "  ...\n"
                  "  [70 71 72 ... 77 78 79]\n"
                  "  [80 81 82 ... 87 88 89]\n"
                  "  [90 91 92 ... 97 98 99]]\n"
                  "\n"
                  " [[100 101 102 ... 107 108 109]\n"
                  "  [110 111 112 ... 117 118 119]\n"
                  "  [120 121 122 ... 127 128 129]\n"
                  "  ...\n  [170 171 172 ... 177 178 179]\n"
                  "  [180 181 182 ... 187 188 189]\n"
                  "  [190 191 192 ... 197 198 199]]\n"
                  "\n"
                  " [[200 201 202 ... 207 208 209]\n"
                  "  [210 211 212 ... 217 218 219]\n"
                  "  [220 221 222 ... 227 228 229]\n"
                  "  ...\n"
                  "  [270 271 272 ... 277 278 279]\n"
                  "  [280 281 282 ... 287 288 289]\n"
                  "  [290 291 292 ... 297 298 299]]\n"
                  "\n"
                  " ...\n"
                  "\n"
                  " [[700 701 702 ... 707 708 709]\n"
                  "  [710 711 712 ... 717 718 719]\n"
                  "  [720 721 722 ... 727 728 729]\n"
                  "  ...\n"
                  "  [770 771 772 ... 777 778 779]\n"
                  "  [780 781 782 ... 787 788 789]\n"
                  "  [790 791 792 ... 797 798 799]]\n"
                  "\n"
                  " [[800 801 802 ... 807 808 809]\n"
                  "  [810 811 812 ... 817 818 819]\n"
                  "  [820 821 822 ... 827 828 829]\n"
                  "  ...\n"
                  "  [870 871 872 ... 877 878 879]\n"
                  "  [880 881 882 ... 887 888 889]\n"
                  "  [890 891 892 ... 897 898 899]]\n"
                  "\n"
                  " [[900 901 902 ... 907 908 909]\n"
                  "  [910 911 912 ... 917 918 919]\n"
                  "  [920 921 922 ... 927 928 929]\n"
                  "  ...\n"
                  "  [970 971 972 ... 977 978 979]\n"
                  "  [980 981 982 ... 987 988 989]\n"
                  "  [990 991 992 ... 997 998 999]]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTemplatePrefix(self):
    # [FUNCTION DOC]
    # Formats a matrix with a template where fixed text precedes the placeholder and asserts the
    # prefix is kept in front of the summarised matrix text. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}", tensor)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTemplatePrefixAndSuffix(self):
    # [FUNCTION DOC]
    # Formats a matrix with fixed text both before and after the placeholder and asserts both
    # pieces surround the summarised matrix text. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}, suffix",
                                               tensor)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]], suffix")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatOneTensorTemplateSuffix(self):
    # [FUNCTION DOC]
    # Formats a matrix with fixed text only after the placeholder and asserts the suffix follows
    # the summarised matrix text. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("{}, suffix", tensor)
      out = self.evaluate(format_output)
      expected = ("[[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]], suffix")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatNoTensor(self):
    # [FUNCTION DOC]
    # Formats a template that has no placeholders using an empty tensor collection and asserts the
    # output equals the template text unchanged. Returns nothing.
    with self.cached_session():
      format_output = string_ops.string_format("No tensor.", ())
      out = self.evaluate(format_output)
      expected = "No tensor."
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatMultiTensor(self):
    # [FUNCTION DOC]
    # Formats two ten-by-ten matrices (the second ten times the first) into a template with two
    # placeholders separated by a comma and newline, passing them as a tuple. Asserts each
    # placeholder is replaced by its own summarised text in order. Returns nothing.
    with self.cached_session():
      tensor_one = array_ops.reshape(math_ops.range(100), [10, 10])
      tensor_two = tensor_one * 10
      format_output = string_ops.string_format("One: {},\nTwo: {}",
                                               (tensor_one, tensor_two))
      out = self.evaluate(format_output)
      expected = ("One: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]],\n"
                  "Two: [[0 10 20 ... 70 80 90]\n"
                  " [100 110 120 ... 170 180 190]\n"
                  " [200 210 220 ... 270 280 290]\n"
                  " ...\n"
                  " [700 710 720 ... 770 780 790]\n"
                  " [800 810 820 ... 870 880 890]\n"
                  " [900 910 920 ... 970 980 990]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatSummarizeOne(self):
    # [FUNCTION DOC]
    # Formats a ten-by-ten matrix with summarize one and asserts only the first and last row and
    # column entries remain, with an elision line between rows. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}", tensor,
                                               summarize=1)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 ... 9]\n"
                  " ...\n"
                  " [90 ... 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatSummarizeTwo(self):
    # [FUNCTION DOC]
    # Formats a ten-by-ten matrix with summarize two and asserts the first two and last two rows
    # and columns are kept, with an elision line between row groups. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: {}", tensor,
                                               summarize=2)
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 ... 8 9]\n"
                  " [10 11 ... 18 19]\n"
                  " ...\n"
                  " [80 81 ... 88 89]\n"
                  " [90 91 ... 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testFormatPlaceholder(self):
    # [FUNCTION DOC]
    # Formats a matrix using a custom placeholder marker instead of the default one and asserts the
    # marker is replaced by the summarised matrix text. Returns nothing.
    with self.cached_session():
      tensor = array_ops.reshape(math_ops.range(100), [10, 10])
      format_output = string_ops.string_format("tensor summary: %t%", tensor,
                                               placeholder="%t%")
      out = self.evaluate(format_output)
      expected = ("tensor summary: [[0 1 2 ... 7 8 9]\n"
                  " [10 11 12 ... 17 18 19]\n"
                  " [20 21 22 ... 27 28 29]\n"
                  " ...\n"
                  " [70 71 72 ... 77 78 79]\n"
                  " [80 81 82 ... 87 88 89]\n"
                  " [90 91 92 ... 97 98 99]]")
      self.assertEqual(compat.as_text(out), expected)

  @test_util.run_in_graph_and_eager_modes()
  def testTensorCountMustMatchPlaceholderCount(self):
    # [FUNCTION DOC]
    # Asserts a value error stating the template expects a different number of tensors than given
    # for three cases: two placeholders with one bare tensor, two placeholders with a one-element
    # list, and one placeholder with a two-element tuple. Returns nothing.
    with self.cached_session():
      with self.assertRaisesRegex(
          ValueError, r"The template expects 2 tensors, but the inputs only has"
          r" 1\.\s.*"):
        tensor = math_ops.range(10)
        format_output = string_ops.string_format("{} {}", tensor)
        self.evaluate(format_output)
    with self.cached_session():
      with self.assertRaisesRegex(
          ValueError, r"The template expects 2 tensors, but the inputs only has"
          r" 1\.\s.*"):
        tensor = math_ops.range(10)
        format_output = string_ops.string_format("{} {}", [tensor])
        self.evaluate(format_output)
    with self.cached_session():
      with self.assertRaisesRegex(
          ValueError, r"The template expects 1 tensors, but the inputs only has"
          r" 2\.\s.*"):
        tensor = math_ops.range(10)
        format_output = string_ops.string_format("{}", (tensor, tensor))
        self.evaluate(format_output)

  @test_util.run_in_graph_and_eager_modes()
  def testTensorAndFormatUnicode(self):
    # [FUNCTION DOC]
    # Formats a scalar string tensor holding an emoji into a template that also contains the emoji
    # and asserts the output keeps the template text and shows the string value in double quotes.
    # Returns nothing.
    with self.cached_session():
      tensor = constant_op.constant("😊")
      format_output = string_ops.string_format("😊:{}", tensor)
      out = self.evaluate(format_output)
      expected = '😊:"😊"'
      self.assertEqual(compat.as_text(out), expected)


if __name__ == "__main__":
  test.main()
