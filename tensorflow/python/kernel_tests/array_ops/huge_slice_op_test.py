# [MODULE DOC]
# Test module for the slice operation on very large tensors (hundreds of millions of elements,
# with offsets beyond the 32-bit range) that need a lot of memory, ideally on a GPU.
# It defines one test case class with one test and has no import-time side effects other than
# imports; it is run through the TensorFlow test runner main entry point when executed as a script.
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
"""Functional tests for slice op that consume a lot of GPU memory."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SliceTest(test.TestCase):
  # [CLASS DOC]
  # Test case holding memory-hungry slice tests. It has no fixtures of its own and relies on the
  # base test case for session handling and assertion helpers.

  def testInt64Slicing(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second half uses the Python 2 only builtin for long integers; under
    # Python 3 that name is undefined, so the test fails with a name error at that point.
    # Test scenario: build a huge boolean tensor by tiling an eight-element alternating
    # false/true pattern about 2^29 times, then slice three elements starting at offset 3 using
    # a 64-bit numpy offset, and again at an offset just above 2^32 given as a 64-bit constant
    # tensor. Expects both slices to equal true, false, true. Runs in a cached session that
    # is forced onto the GPU when one is available. Takes no inputs and returns nothing.
    with self.cached_session(force_gpu=test.is_gpu_available()):
      a_large = array_ops.tile(
          constant_op.constant(np.array([False, True] * 4)), [2**29 + 3])
      slice_t = array_ops.slice(a_large, np.asarray([3]).astype(np.int64), [3])
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([True, False, True], slice_val)

      slice_t = array_ops.slice(
          a_large, constant_op.constant([long(2)**32 + 3], dtype=dtypes.int64),
          [3])
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([True, False, True], slice_val)
