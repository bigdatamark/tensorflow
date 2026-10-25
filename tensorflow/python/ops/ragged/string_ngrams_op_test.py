# [MODULE DOC]
# Unit tests for the string n-grams op in the ragged string ops module. Covers single and multiple
# n-gram widths, padding with left/right pad values, padding size caps, short-sequence
# preservation, ragged, dense, list and vector inputs, empty input, argument validation errors and
# the unknown-rank error inside a traced function. Run through the test runner entry point.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the b"License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an b"AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the Tensorflow strings.ngrams op."""

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test


class StringNgramsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the ragged string n-grams op. Most tests build string data (ragged, dense, list
  # or vector), call the op with a width, separator and optional padding settings, evaluate the
  # result and compare it with expected joined strings. Stateless; uses parameterized testing.

  def test_unpadded_ngrams(self):
    # [FUNCTION DOC]
    # Ragged strings with trigram width and a bar separator, no padding. Expects only full
    # trigrams; the row with two items yields an empty list.
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb|cc", b"bb|cc|dd"], []]
    self.assertAllEqual(expected_ngrams, result)

  def test_tuple_multi_ngrams(self):
    # [FUNCTION DOC]
    # Width given as a tuple of two and three. Expects, per row, all bigrams followed by all
    # trigrams in the order of the widths.
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(2, 3), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb", b"bb|cc", b"cc|dd", b"aa|bb|cc", b"bb|cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_tuple_multi_ngrams_inverted_order(self):
    # [FUNCTION DOC]
    # Width tuple given as three then two. Expects trigrams first, then bigrams, showing output
    # follows the order of the widths rather than sorted order.
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(3, 2), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb|cc", b"bb|cc|dd", b"aa|bb", b"bb|cc", b"cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_list_multi_ngrams(self):
    # [FUNCTION DOC]
    # Same as the tuple test but the widths (two, three) are given as a list. Expects identical
    # results: bigrams then trigrams per row.
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=[2, 3], separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb", b"bb|cc", b"cc|dd", b"aa|bb|cc", b"bb|cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_multi_ngram_ordering(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: effectively duplicates the inverted-order tuple test, only using a list.
    # Widths given as a list of three then two. Expects trigrams before bigrams in each row.
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=[3, 2], separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb|cc", b"bb|cc|dd", b"aa|bb", b"bb|cc", b"cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_fully_padded_ngrams(self):
    # [FUNCTION DOC]
    # Trigram width with a pair of left and right pad strings and no explicit padding width.
    # Expects each row to be padded with two pad items on each side, producing sequence length
    # plus two n-grams per row.
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|RP", b"a|RP|RP"],  # 0
        [b"LP|LP|b", b"LP|b|c", b"b|c|d", b"c|d|RP", b"d|RP|RP"],  # 1
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"]  # 2
    ]
    self.assertAllEqual(expected_ngrams, result)

  def test_ngram_padding_size_cap(self):
    # [FUNCTION DOC]
    # Requests a padding width of ten with trigrams. Expects the result to be identical to the
    # fully padded case because padding is capped at the n-gram width minus one.
    # Validate that the padding size is never greater than ngram_size - 1.
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=3,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=10)
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|RP", b"a|RP|RP"],  # 0
        [b"LP|LP|b", b"LP|b|c", b"b|c|d", b"c|d|RP", b"d|RP|RP"],  # 1
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"]  # 2
    ]
    self.assertAllEqual(expected_ngrams, result)

  def test_singly_padded_ngrams(self):
    # [FUNCTION DOC]
    # Width five with padding width one. Only the three-item row is long enough once padded, so
    # expects a single padded five-gram for it and empty lists for the other rows.
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=1)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[], [b"LP|b|c|d|RP"], []]
    self.assertAllEqual(expected_ngrams, result)

  def test_singly_padded_ngrams_with_preserve_short(self):
    # [FUNCTION DOC]
    # Same data as the singly padded test but short sequences are preserved. Rows too short for a
    # five-gram are emitted as one n-gram made of the whole padded row.
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=1,
        preserve_short_sequences=True)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"LP|a|RP"], [b"LP|b|c|d|RP"], [b"LP|e|f|RP"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_singly_padded_multiple_ngrams(self):
    # [FUNCTION DOC]
    # Widths one and five with padding width one. Expects unigrams for every row plus the
    # five-gram only where the padded row is long enough.
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=(1, 5),
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=1)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"a"], [b"b", b"c", b"d", b"LP|b|c|d|RP"], [b"e", b"f"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_single_padding_string(self):
    # [FUNCTION DOC]
    # Passes one pad string instead of a pair; it is used on both sides. Expects the same
    # behavior as the singly padded test with that string as the pad.
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=b"[PAD]",
        padding_width=1)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[], [b"[PAD]|b|c|d|[PAD]"], []]
    self.assertAllEqual(expected_ngrams, result)

  def test_explicit_multiply_padded_ngrams(self):
    # [FUNCTION DOC]
    # One single-item row, width five, padding width two. Expects one n-gram consisting of two
    # left pads, the item and two right pads.
    data = [[b"a"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=2)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"LP|LP|a|RP|RP"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions(self):
    # [FUNCTION DOC]
    # Four-level nested ragged input with trigrams. Expects n-grams computed along the innermost
    # dimension while the outer nesting is preserved; short inner rows give empty lists.
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb|cc", b"bb|cc|dd"]], [[]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions_and_preserve(self):
    # [FUNCTION DOC]
    # Same nested input with trigrams and short-sequence preservation. Expects the short inner
    # row to yield its whole sequence joined as a single n-gram.
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=3,
        separator=b"|",
        preserve_short_sequences=True)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb|cc", b"bb|cc|dd"]], [[b"ee|ff"]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions_bigrams(self):
    # [FUNCTION DOC]
    # Same nested input with bigrams. Expects all adjacent pairs in the innermost rows, nesting
    # preserved.
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=2, separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb", b"bb|cc", b"cc|dd"]], [[b"ee|ff"]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions_and_multiple_ngrams(
      self):
    # [FUNCTION DOC]
    # Same nested input with widths three and four. Expects trigrams followed by the four-gram
    # for the long inner row and an empty list for the short one.
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(3, 4), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb|cc", b"bb|cc|dd", b"aa|bb|cc|dd"]], [[]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_dense_input_rank_3(self):
    # [FUNCTION DOC]
    # Dense rank-three string tensor with trigrams and pad pair. Expects n-grams along the last
    # axis (including ones containing empty strings) and that the result is a dense tensor, not
    # a ragged one.
    data = [[[b"a", b"z"], [b"b", b""]], [[b"b", b""], [b"e", b"f"]]]
    data_tensor = constant_op.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"],
                        [b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"]],
                       [[b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"],
                        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"]]]
    self.assertIsInstance(ngram_op, tensor.Tensor)
    self.assertAllEqual(expected_ngrams, result)

  def test_dense_input(self):
    # [FUNCTION DOC]
    # Dense rank-two string tensor with trigrams and pad pair. Expects padded n-grams per row and
    # a dense tensor result.
    data = [[b"a", b"z"], [b"b", b""], [b"e", b"f"]]
    data_tensor = constant_op.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"],
        [b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"],
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"],
    ]
    self.assertIsInstance(ngram_op, tensor.Tensor)
    self.assertAllEqual(expected_ngrams, result)

  def test_input_list_input(self):
    # [FUNCTION DOC]
    # Passes a plain nested list (not a tensor) to the op with padding. Expects the same result
    # as the dense input test, showing automatic conversion.
    data = [[b"a", b"z"], [b"b", b""], [b"e", b"f"]]
    ngram_op = ragged_string_ops.ngrams(
        data, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"],
        [b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"],
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"],
    ]
    self.assertAllEqual(expected_ngrams, result)

  def test_vector_input(self):
    # [FUNCTION DOC]
    # Rank-one input of two strings with trigrams and padding. Expects a single flat list of
    # padded n-grams for the whole vector.
    data = [b"a", b"z"]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"]
    self.assertAllEqual(expected_ngrams, result)

  def test_dense_input_with_multiple_ngrams(self):
    # [FUNCTION DOC]
    # Rectangular data (built via the ragged constant factory, despite the name) with widths one,
    # two and three, no padding. Expects per row unigrams, then bigrams, then trigrams.
    data = [[b"a", b"b", b"c", b"d"], [b"e", b"f", b"g", b"h"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(1, 2, 3), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[
        b"a", b"b", b"c", b"d", b"a|b", b"b|c", b"c|d", b"a|b|c", b"b|c|d"
    ], [b"e", b"f", b"g", b"h", b"e|f", b"f|g", b"g|h", b"e|f|g", b"f|g|h"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_input_with_no_values(self):
    # [FUNCTION DOC]
    # Three empty rows of string type with widths one and two. Expects row splits of all zeros
    # and an empty string values tensor.
    data = ragged_factory_ops.constant([[], [], []], dtype=dtypes.string)
    ngram_op = ragged_string_ops.ngrams(data, (1, 2))
    result = self.evaluate(ngram_op)
    self.assertAllEqual([0, 0, 0, 0], result.row_splits)
    self.assertAllEqual(constant_op.constant([], dtype=dtypes.string),
                        result.values)

  @parameterized.parameters([
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          pad_values=5,
          exception=TypeError,
          error="pad_values must be a string, tuple of strings, or None."),
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          pad_values=[5, 3],
          exception=TypeError,
          error="pad_values must be a string, tuple of strings, or None."),
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          padding_width=0,
          pad_values="X",
          error="padding_width must be greater than 0."),
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          padding_width=1,
          error="pad_values must be provided if padding_width is set."),
      dict(
          data=b"hello",
          ngram_width=2,
          padding_width=1,
          pad_values="X",
          error="Data must have rank>0"),
      dict(
          data=[b"hello", b"world"],
          ngram_width=[1, 2, -1],
          padding_width=1,
          pad_values="X",
          error="All ngram_widths must be greater than 0. Got .*"),
  ])
  def test_error(self,
                 data,
                 ngram_width,
                 separator=" ",
                 pad_values=None,
                 padding_width=None,
                 preserve_short_sequences=False,
                 error=None,
                 exception=ValueError):
    # [FUNCTION DOC]
    # Parameterized error test. Each case supplies data, width, optional separator, pad values,
    # padding width, preserve flag, a message pattern and an exception type (default value error;
    # the pad value type cases use type error). Calls the op and asserts that it raises the given
    # exception with a message matching the pattern. Cases: bad pad value types, padding width of
    # zero, padding width without pad values, scalar data, and a nonpositive n-gram width.
    with self.assertRaisesRegex(exception, error):
      ragged_string_ops.ngrams(data, ngram_width, separator, pad_values,
                               padding_width, preserve_short_sequences)

  def test_unknown_rank_error(self):
    # [FUNCTION DOC]
    # Traces a function whose input signature has unknown rank and calls the op on it. Expects a
    # value error saying the rank of data must be known when the function is called.
    # Use a tf.function that erases shape information.
    @def_function.function(
        input_signature=[tensor.TensorSpec(None, dtypes.string)])
    def f(v):
      # [FUNCTION DOC]
      # Traced helper taking a string tensor of unknown shape and returning its bigrams via the
      # ngrams op; tracing triggers the unknown-rank error the test expects.
      return ragged_string_ops.ngrams(v, 2)

    with self.assertRaisesRegex(ValueError, "Rank of data must be known."):
      f([b"foo", b"bar"])


if __name__ == "__main__":
  test.main()
