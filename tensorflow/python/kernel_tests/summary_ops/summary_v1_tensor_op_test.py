# [MODULE DOC]
# Tests for the legacy (v1) tensor summary op: tags under name scopes, value round trips for
# scalars, strings, arrays, booleans, and display name / description metadata handling.

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BAvSIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for summary V1 tensor op."""

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib


class SummaryV1TensorOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy tensor summary op: tags, scalar, string, array and boolean values, and
  # metadata (display name and description).

  def _SummarySingleValue(self, s):
    # [FUNCTION DOC]
    # HELPER: parses a serialized summary, asserts it has exactly one value and returns that value.
    summ = summary_pb2.Summary()
    summ.ParseFromString(s)
    self.assertEqual(len(summ.value), 1)
    return summ.value[0]

  def _AssertNumpyEq(self, actual, expected):
    # [FUNCTION DOC]
    # HELPER: asserts two array-likes are exactly equal element-wise and same shape.
    self.assertTrue(np.array_equal(actual, expected))

  def testTags(self):
    # [FUNCTION DOC]
    # Creates tensor summaries at top level and inside nested name scopes, including one using the
    # default name. Asserts the tags carry the name scope prefixes.
    with self.cached_session() as sess:
      c = constant_op.constant(1)
      s1 = summary_lib.tensor_summary("s1", c)
      with ops.name_scope("foo", skip_on_eager=False):
        s2 = summary_lib.tensor_summary("s2", c)
        with ops.name_scope("zod", skip_on_eager=False):
          s3 = summary_lib.tensor_summary("s3", c)
          s4 = summary_lib.tensor_summary("TensorSummary", c)
      summ1, summ2, summ3, summ4 = self.evaluate([s1, s2, s3, s4])

    v1 = self._SummarySingleValue(summ1)
    self.assertEqual(v1.tag, "s1")

    v2 = self._SummarySingleValue(summ2)
    self.assertEqual(v2.tag, "foo/s2")

    v3 = self._SummarySingleValue(summ3)
    self.assertEqual(v3.tag, "foo/zod/s3")

    v4 = self._SummarySingleValue(summ4)
    self.assertEqual(v4.tag, "foo/zod/TensorSummary")

  def testScalarSummary(self):
    # [FUNCTION DOC]
    # Summarizes a float constant and asserts the decoded tensor equals ten.
    with self.cached_session() as sess:
      const = constant_op.constant(10.0)
      summ = summary_lib.tensor_summary("foo", const)
      result = self.evaluate(summ)

    value = self._SummarySingleValue(result)
    n = tensor_util.MakeNdarray(value.tensor)
    self._AssertNumpyEq(n, 10)

  def testStringSummary(self):
    # [FUNCTION DOC]
    # Summarizes a byte-string constant and asserts the decoded tensor equals the string.
    s = b"foobar"
    with self.cached_session() as sess:
      const = constant_op.constant(s)
      summ = summary_lib.tensor_summary("foo", const)
      result = self.evaluate(summ)

    value = self._SummarySingleValue(result)
    n = tensor_util.MakeNdarray(value.tensor)
    self._AssertNumpyEq(n, s)

  def testManyScalarSummary(self):
    # [FUNCTION DOC]
    # Summarizes a five-by-five-by-five tensor of ones and asserts it decodes to the same array.
    with self.cached_session() as sess:
      const = array_ops.ones([5, 5, 5])
      summ = summary_lib.tensor_summary("foo", const)
      result = self.evaluate(summ)
    value = self._SummarySingleValue(result)
    n = tensor_util.MakeNdarray(value.tensor)
    self._AssertNumpyEq(n, np.ones([5, 5, 5]))

  def testManyStringSummary(self):
    # [FUNCTION DOC]
    # Summarizes a two-by-two matrix of byte strings and asserts it decodes unchanged.
    strings = [[b"foo bar", b"baz"], [b"zoink", b"zod"]]
    with self.cached_session() as sess:
      const = constant_op.constant(strings)
      summ = summary_lib.tensor_summary("foo", const)
      result = self.evaluate(summ)
    value = self._SummarySingleValue(result)
    n = tensor_util.MakeNdarray(value.tensor)
    self._AssertNumpyEq(n, strings)

  def testManyBools(self):
    # [FUNCTION DOC]
    # Summarizes a list of six booleans and asserts it decodes unchanged.
    bools = [True, True, True, False, False, False]
    with self.cached_session() as sess:
      const = constant_op.constant(bools)
      summ = summary_lib.tensor_summary("foo", const)
      result = self.evaluate(summ)

    value = self._SummarySingleValue(result)
    n = tensor_util.MakeNdarray(value.tensor)
    self._AssertNumpyEq(n, bools)

  def testSummaryDescriptionAndDisplayName(self):
    # [FUNCTION DOC]
    # Checks metadata: defaults are empty; display name and description can come from arguments or
    # from a metadata message; when both are supplied the explicit arguments override the message.
    with self.cached_session() as sess:

      def get_description(summary_op):
        # [FUNCTION DOC]
        # HELPER: evaluates the given summary op, parses the result and returns the metadata of its first
        # value.
        summ_str = self.evaluate(summary_op)
        summ = summary_pb2.Summary()
        summ.ParseFromString(summ_str)
        return summ.value[0].metadata

      const = constant_op.constant(1)
      # Default case; no description or display name
      simple_summary = summary_lib.tensor_summary("simple", const)

      descr = get_description(simple_summary)
      self.assertEqual(descr.display_name, "")
      self.assertEqual(descr.summary_description, "")

      # Values are provided via function args
      with_values = summary_lib.tensor_summary(
          "simple",
          const,
          display_name="my name",
          summary_description="my description")

      descr = get_description(with_values)
      self.assertEqual(descr.display_name, "my name")
      self.assertEqual(descr.summary_description, "my description")

      # Values are provided via the SummaryMetadata arg
      metadata = summary_pb2.SummaryMetadata()
      metadata.display_name = "my name"
      metadata.summary_description = "my description"

      with_metadata = summary_lib.tensor_summary(
          "simple", const, summary_metadata=metadata)
      descr = get_description(with_metadata)
      self.assertEqual(descr.display_name, "my name")
      self.assertEqual(descr.summary_description, "my description")

      # If both SummaryMetadata and explicit args are provided, the args win
      overwrite = summary_lib.tensor_summary(
          "simple",
          const,
          summary_metadata=metadata,
          display_name="overwritten",
          summary_description="overwritten")
      descr = get_description(overwrite)
      self.assertEqual(descr.display_name, "overwritten")
      self.assertEqual(descr.summary_description, "overwritten")


if __name__ == "__main__":
  test.main()
