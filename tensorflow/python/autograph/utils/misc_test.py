# [MODULE DOC]
# Unit tests for the misc utility module (alias_tensors and get_range_len). One test case class;
# some tests run in deprecated graph mode with a cached session. No import-time side effects;
# runs the TF test main when executed directly.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for misc module."""

from tensorflow.python.autograph.utils import misc
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops.variables import Variable
from tensorflow.python.platform import test


class MiscTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the misc module; no shared fixtures.

  @test_util.run_deprecated_v1
  def test_alias_single_tensor(self):
    # [FUNCTION DOC]
    # Aliases one constant tensor, asserts the result is a different object and that evaluating it
    # in a session gives the original value 1. Runs in deprecated v1 graph mode.
    a = constant(1)

    new_a = misc.alias_tensors(a)
    self.assertFalse(new_a is a)
    with self.cached_session() as sess:
      self.assertEqual(1, self.evaluate(new_a))

  @test_util.run_deprecated_v1
  def test_alias_tensors(self):
    # [FUNCTION DOC]
    # Aliases a tensor, a variable, a string and a list together and unpacks the result.
    # Asserts the tensor is replaced by a new object (that evaluates to 1) while the variable,
    # string and list are returned as the identical objects. Runs in deprecated v1 graph mode.
    a = constant(1)
    v = Variable(2)
    s = 'a'
    l = [1, 2, 3]

    new_a, new_v, new_s, new_l = misc.alias_tensors(a, v, s, l)

    self.assertFalse(new_a is a)
    self.assertTrue(new_v is v)
    self.assertTrue(new_s is s)
    self.assertTrue(new_l is l)
    with self.cached_session() as sess:
      self.assertEqual(1, self.evaluate(new_a))

  def test_get_range_len(self):
    # [FUNCTION DOC]
    # Wraps the range-length function as a graph function, then for every combination of start,
    # limit and step values from -3 to 2 as tensors (skipping a zero step) computes the length.
    # Asserts each equals the length of the corresponding Python range.
    get_range_as_graph = def_function.function(misc.get_range_len)
    test_range = [(i, constant_op.constant(i)) for i in range(-3, 3)]
    results = []
    for i, ti in test_range:
      for j, tj in test_range:
        for k, tk in test_range:
          if k == 0:
            continue
          results.append(((i, j, k), get_range_as_graph(ti, tj, tk)))

    for (i, j, k), result_tensor in results:
      self.assertEqual(
          len(list(range(i, j, k))), self.evaluate(result_tensor))


if __name__ == '__main__':
  test.main()
