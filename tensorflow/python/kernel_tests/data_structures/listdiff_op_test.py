# [MODULE DOC]
# Tests for the list-difference operation, exposed as the one-dimensional set-difference function
# in array_ops. A single test class feeds small lists and random lists through the op for several
# element types and both index integer types and checks the returned values and indices.
# Importing has no side effects; running as a script starts the TensorFlow test runner.
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
"""Tests for tensorflow.kernels.listdiff_op."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

_TYPES = [
    dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.string
]


class ListDiffTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the list-difference op. Each test supplies two lists, the expected values that are
  # in the first list but not in the second (order and duplicates preserved) and the expected
  # positions of those values in the first list, and delegates to a shared helper that runs all
  # supported element types and index types. Has no instance state.

  def _testListDiff(self, x, y, out, idx):
    # [FUNCTION DOC]
    # HELPER: shared checker used by every test in this class.
    # KNOWN ISSUE: when the string type is reached, the local x, y and out lists are rebound to
    # byte-string versions for the rest of the call; because string is the last type this is harmless
    # now, but reordering the types would feed byte strings to later numeric types.
    # Inputs: x, the list to take values from; y, the list of values to remove; out, the expected
    # remaining values; idx, the expected positions of the remaining values in x.
    # For each element type among int32, int64, float32, float64 and string, and for each index type
    # (int32 and int64), converts x and y to tensors, runs the list-difference op in a cached
    # session, and asserts the output values and indices equal the expectations and that both outputs
    # have rank one. For the string type the inputs and expected values are first converted to byte
    # strings. Returns nothing; raises assertion failures on mismatch.
    for dtype in _TYPES:
      if dtype == dtypes.string:
        x = [compat.as_bytes(str(a)) for a in x]
        y = [compat.as_bytes(str(a)) for a in y]
        out = [compat.as_bytes(str(a)) for a in out]
      for diff_func in [array_ops.setdiff1d]:
        for index_dtype in [dtypes.int32, dtypes.int64]:
          with self.cached_session() as sess:
            x_tensor = ops.convert_to_tensor(x, dtype=dtype)
            y_tensor = ops.convert_to_tensor(y, dtype=dtype)
            out_tensor, idx_tensor = diff_func(x_tensor, y_tensor,
                                               index_dtype=index_dtype)
            tf_out, tf_idx = self.evaluate([out_tensor, idx_tensor])
          self.assertAllEqual(tf_out, out)
          self.assertAllEqual(tf_idx, idx)
          self.assertEqual(1, out_tensor.get_shape().ndims)
          self.assertEqual(1, idx_tensor.get_shape().ndims)

  def testBasic1(self):
    # [FUNCTION DOC]
    # Removes the first two values from a list of four; expects the last two values at positions two
    # and three.
    x = [1, 2, 3, 4]
    y = [1, 2]
    out = [3, 4]
    idx = [2, 3]
    self._testListDiff(x, y, out, idx)

  def testBasic2(self):
    # [FUNCTION DOC]
    # Removes a single middle value from a list of four; expects the other three values with their
    # original positions.
    x = [1, 2, 3, 4]
    y = [2]
    out = [1, 3, 4]
    idx = [0, 2, 3]
    self._testListDiff(x, y, out, idx)

  def testBasic3(self):
    # [FUNCTION DOC]
    # Removes two unordered values from an unordered list; expects the remaining values keep their
    # original order and positions.
    x = [1, 4, 3, 2]
    y = [4, 2]
    out = [1, 3]
    idx = [0, 2]
    self._testListDiff(x, y, out, idx)

  def testDuplicates(self):
    # [FUNCTION DOC]
    # List with repeated values; removes some values and expects every remaining duplicate to be
    # kept in original order with its own position.
    x = [1, 2, 4, 3, 2, 3, 3, 1]
    y = [4, 2]
    out = [1, 3, 3, 3, 1]
    idx = [0, 3, 5, 6, 7]
    self._testListDiff(x, y, out, idx)

  def testRandom(self):
    # [FUNCTION DOC]
    # Runs ten random trials. Each draws two random integer lists (length up to fifty, values in a
    # small range) using the numpy random generator, computes the expected output and positions in
    # plain Python, and runs the shared checker. The random seed is not fixed.
    num_random_tests = 10
    int_low = -7
    int_high = 8
    max_size = 50
    for _ in range(num_random_tests):
      x_size = np.random.randint(max_size + 1)
      x = np.random.randint(int_low, int_high, size=x_size)
      y_size = np.random.randint(max_size + 1)
      y = np.random.randint(int_low, int_high, size=y_size)
      out_idx = [(entry, pos) for pos, entry in enumerate(x) if entry not in y]
      if out_idx:
        out, idx = map(list, zip(*out_idx))
      else:
        out = []
        idx = []
      self._testListDiff(list(x), list(y), out, idx)

  def testFullyOverlapping(self):
    # [FUNCTION DOC]
    # Both lists are identical; expects empty outputs.
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    out = []
    idx = []
    self._testListDiff(x, y, out, idx)

  def testNonOverlapping(self):
    # [FUNCTION DOC]
    # The lists share no values; expects the first list unchanged with positions zero to three.
    x = [1, 2, 3, 4]
    y = [5, 6]
    out = x
    idx = np.arange(len(x))
    self._testListDiff(x, y, out, idx)

  def testEmptyX(self):
    # [FUNCTION DOC]
    # First list empty; expects empty outputs.
    x = []
    y = [1, 2]
    out = []
    idx = []
    self._testListDiff(x, y, out, idx)

  def testEmptyY(self):
    # [FUNCTION DOC]
    # Second list empty; expects the first list unchanged with all positions.
    x = [1, 2, 3, 4]
    y = []
    out = x
    idx = np.arange(len(x))
    self._testListDiff(x, y, out, idx)

  def testEmptyXY(self):
    # [FUNCTION DOC]
    # Both lists empty; expects empty outputs.
    x = []
    y = []
    out = []
    idx = []
    self._testListDiff(x, y, out, idx)


if __name__ == "__main__":
  test.main()
