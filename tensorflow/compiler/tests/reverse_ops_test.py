# [MODULE DOC]
# XLA compiler tests for the reverse op on a 4-D tensor: reversing single dimensions (positive
# and negative indices) and combinations of dimensions, compared against numpy slicing. No
# import-time side effects.
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
"""Functional tests for XLA Reverse Ops."""

import itertools

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class ReverseOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the reverse op on the XLA test device. Tests choose axes to reverse on a
  # 7x5x9x11 tensor and delegate verification to a shared helper.

  def testReverseOneDim(self):
    # [FUNCTION DOC]
    # Reverses each single axis of a 7x5x9x11 tensor, using every index from -4 to 3 (negative
    # indices count from the end), checking each against numpy.
    shape = (7, 5, 9, 11)
    for revdim in range(-len(shape), len(shape)):
      self._AssertReverseEqual([revdim], shape)

  def testReverseMoreThanOneDim(self):
    # [FUNCTION DOC]
    # Reverses combinations of 2 to 4 axes of a 7x5x9x11 tensor, using index sets shifted by an
    # offset so that mixes of negative and positive indices are covered without referring to the
    # same axis twice; checks each combination against numpy.
    shape = (7, 5, 9, 11)
    # The offset is used to test various (but not all) combinations of negative
    # and positive axis indices that are guaranteed to not collide at the same
    # index.
    for revdims in itertools.chain.from_iterable(
        itertools.combinations(range(-offset,
                                     len(shape) - offset), k)
        for k in range(2,
                       len(shape) + 1)
        for offset in range(0, len(shape))):
      self._AssertReverseEqual(revdims, shape)

  def _AssertReverseEqual(self, revdims, shape):
    # [FUNCTION DOC]
    # HELPER: shared reverse check. Seeds the global numpy random generator (side effect) with
    # 120, builds random integer-valued data of the given shape, runs the reverse op on the test
    # device with the list of axes revdims, and asserts the flattened result equals the numpy
    # reversal along those axes (axes matched whether given as positive or negative indices).
    # KNOWN ISSUE: the data is created as floating point values but fed to an int32 placeholder;
    # this works only because the values are whole numbers. Returns nothing.
    np.random.seed(120)
    pval = np.random.randint(0, 100, size=shape).astype(float)
    with self.session():
      with self.test_scope():
        p = array_ops.placeholder(dtypes.int32, shape=shape)
        axis = constant_op.constant(
            np.array(revdims, dtype=np.int32),
            shape=(len(revdims),),
            dtype=dtypes.int32)
        rval = array_ops.reverse(p, axis).eval({p: pval})

        slices = tuple(
            slice(-1, None, -1)
            if d in revdims or d - len(shape) in revdims else slice(None)
            for d in range(len(shape))
        )
      self.assertEqual(pval[slices].flatten().tolist(), rval.flatten().tolist())


if __name__ == '__main__':
  googletest.main()
