# [MODULE DOC]
# Tests for the sparse mask operation applied to indexed slices (array_ops.sparse_mask).
# Defines a single test case class that is run through the standard TensorFlow test main
# entry point. No import-time side effects beyond imports. Not a helper module.
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

import numpy as np

from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SparseMaskTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the sparse mask operation on indexed slices. It has a single test
  # method and no shared fixtures; it relies on a cached session from the base test class.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Builds an indexed-slices object from a random 4x4 float matrix with row indices 0, 2, 3, 4,
    # then masks out index 0 using the sparse mask operation inside a cached session.
    # Asserts the resulting values equal the last three rows of the input and the resulting
    # indices equal 2, 3, 4. Runs only in graph mode (decorated as deprecated-v1). Returns nothing.
    values = np.random.rand(4, 4).astype(np.single)
    indices = np.array([0, 2, 3, 4], dtype=np.int32)
    mask_indices = np.array([0], dtype=np.int32)

    out_values = values[1:, :]
    out_indices = np.array([2, 3, 4], dtype=np.int32)

    with self.cached_session() as sess:
      values_tensor = ops.convert_to_tensor(values)
      indices_tensor = ops.convert_to_tensor(indices)
      mask_indices_tensor = ops.convert_to_tensor(mask_indices)

      t = indexed_slices.IndexedSlices(values_tensor, indices_tensor)
      masked_t = array_ops.sparse_mask(t, mask_indices_tensor)

      tf_out_values, tf_out_indices = sess.run(
          [masked_t.values, masked_t.indices])

      self.assertAllEqual(tf_out_values, out_values)
      self.assertAllEqual(tf_out_indices, out_indices)


if __name__ == "__main__":
  test.main()
