# [MODULE DOC]
# Unit test module for the autograph operations helper module; it checks that wrapping a Python
# callable as a graph operation runs the callable for its side effects and yields a constant
# integer one, for both plain and tensor arguments. No import-time side effects.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for autograph_ops module."""

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import autograph_ops
from tensorflow.python.platform import test


class AutographOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Python-function wrapping helper of the autograph operations module. No
  # shared fixtures.

  def test_wrap_py_func_dummy_return(self):
    # [FUNCTION DOC]
    # Test: wraps a callable that increments a counter list as a side effect, first with a plain
    # integer argument and then with a constant tensor argument, inside a cached session. After
    # each evaluation it asserts the result equals one (the dummy return value) and that the
    # counter went up by exactly one, proving the callable ran once per evaluation. No inputs, no
    # return value.
    side_counter = [0]

    def test_fn(_):
      # [FUNCTION DOC]
      # HELPER: callable under test; ignores its argument and increments the shared counter by
      # one as a side effect. Returns nothing.
      side_counter[0] += 1

    with self.cached_session():
      result = autograph_ops.wrap_py_func(test_fn, (5,))
      self.assertEqual(1, self.evaluate(result))
      self.assertEqual([1], side_counter)
      result = autograph_ops.wrap_py_func(test_fn, (constant_op.constant(5),))
      self.assertEqual(1, self.evaluate(result))
      self.assertEqual([2], side_counter)


if __name__ == '__main__':
  test.main()
