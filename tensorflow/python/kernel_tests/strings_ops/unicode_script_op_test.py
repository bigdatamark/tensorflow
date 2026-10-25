# [MODULE DOC]
# Tests and a benchmark for the unicode script op (string_ops.unicode_script), which maps each
# integer codepoint to its numeric Unicode script code, or minus one for invalid codepoints.
# Test-only module with no import-time side effects.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
"""Functional tests for UnicodeScript op."""

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class UnicodeScriptOpTest(test.TestCase):
  # [CLASS DOC]
  # Functional tests for the unicode script op, checking script codes for valid and invalid
  # codepoints in a cached graph session.

  @test_util.run_deprecated_v1
  def testValidScripts(self):
    # [FUNCTION DOC]
    # Feeds four codepoints (Latin letter, Cyrillic letter, Han ideograph, comma) and asserts
    # the script codes for Latin, Cyrillic, Han and Common are returned in order. Runs in
    # graph mode version one.
    inputs = [
        ord("a"),
        0x0411,  # CYRILLIC CAPITAL LETTER BE
        0x82b8,  # CJK UNIFIED IDEOGRAPH-82B8
        ord(",")
    ]
    with self.cached_session():
      input_vector = constant_op.constant(inputs, dtypes.int32)
      outputs = string_ops.unicode_script(input_vector).eval()
      self.assertAllEqual(
          outputs,
          [
              25,  # USCRIPT_LATIN (LATN)
              8,  # USCRIPT_CYRILLIC (CYRL)
              17,  # USCRIPT_HAN (HANI)
              0  # USCRIPT_COMMON (ZYYY)
          ])

  @test_util.run_deprecated_v1
  def testInvalidScript(self):
    # [FUNCTION DOC]
    # Feeds a negative codepoint and one beyond the Unicode range. Asserts both map to minus
    # one. Runs in graph mode version one.
    inputs = [-100, 0xffffff]
    with self.cached_session():
      input_vector = constant_op.constant(inputs, dtypes.int32)
      outputs = string_ops.unicode_script(input_vector).eval()
      self.assertAllEqual(outputs, [-1, -1])


class UnicodeScriptBenchmarks(test.Benchmark):
  # [CLASS DOC]
  # Benchmark for the unicode script op on a large generated list of codepoints.

  # Generate some random-ish input by jumping around in unicode characters
  def _generateBenchmarkInput(self, size):
    # [FUNCTION DOC]
    # HELPER: produces input for the script benchmark.
    # Builds a list of the requested number of codepoints starting at the letter a, stepping by
    # one and jumping ahead by one hundred every twenty characters, wrapping back to the start
    # once past a fixed upper bound. Returns the list; no side effects.
    chars = []
    i = 0
    offset = 0
    continuity_size = 20
    while i < size:
      chars.append(ord("a") + offset)
      i += 1
      offset += 1
      if i % continuity_size == 0:
        offset += 100
        if offset > 0x1F940:
          offset = 0

    return chars

  def benchmark_unicode_script(self):
    # [FUNCTION DOC]
    # Generates one million codepoints, builds the script op on them and runs the op benchmark
    # helper with at least one hundred iterations in a session using the benchmark config.
    # Reports timing through the benchmark framework; returns nothing.
    with session.Session(config=benchmark.benchmark_config()) as sess:
      chars = self._generateBenchmarkInput(1000000)
      script = string_ops.unicode_script(chars)
      self.run_op_benchmark(sess, script.op, min_iters=100)


if __name__ == "__main__":
  test.main()
