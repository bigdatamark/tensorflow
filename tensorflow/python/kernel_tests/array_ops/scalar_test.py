# [MODULE DOC]
# KNOWN ISSUE: the source itself carries a TODO saying the whole test should be removed once old
# graph versions are no longer supported; it covers deprecated behavior.
# Tests for legacy scalar strictness versus leniency at different graph producer versions. For a
# variety of ops (concat, fill, pad, random, reshape, sharded filename/filespec, unsorted segment
# sum, range, slice, sparse-to-dense, tile) it feeds scalar arguments where vectors are expected
# and checks that new graph versions raise an error while legacy versions (where listed) still
# compute the old result. Importing the nn gradient module has a registration side effect.
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
"""Tests for scalar strictness and scalar leniency."""

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


# TODO(rmlarsen) : Remove this test completely after we stop supporting GraphDef
# version 5 and remove support of legacy scalars from Concat, Fill, Range,
# and Reshape.
class ScalarTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying scalar strictness across graph producer versions. A generic check helper
  # builds the op from placeholders for each producer version and asserts error or legacy result;
  # each test method supplies one op with scalar-versus-vector argument combinations.

  def check(self, op, args, error, correct=None, lenient=None, strict=None):
    # [FUNCTION DOC]
    # HELPER: shared check used by every test method. Inputs: op callable, args tuple of argument
    # values (nested tuples allowed), error pattern expected for strict versions, correct expected
    # result for lenient versions (optional), lenient list of graph producer versions that still
    # accept the scalar (default empty) and strict list of versions that must fail (default five
    # and six). For each version it builds a fresh graph with that producer version, replaces each
    # argument by a placeholder fed with its value (so shape inference is bypassed), applies the op
    # and runs it: strict versions must raise an op error matching the pattern, lenient ones must
    # run and, if correct is given, equal it. Returns nothing.
    # KNOWN ISSUE: the argument evaluation inside uses an implicit default session which is not
    # set up at that point, and with the default lenient list empty the expected result is never
    # compared for tests that do not pass lenient versions (concat, pad, slice, ...).
    if lenient is None:
      lenient = []
    if strict is None:
      strict = [5, 6]
    # Use placeholders to bypass shape inference, since only the C++
    # GraphDef level is ever scalar lenient.
    def placeholders(args, feed):
      # [FUNCTION DOC]
      # HELPER: recursively converts argument values into placeholders. For a tuple it returns a
      # list of converted entries; for any other value it evaluates it as a constant, creates a
      # placeholder of the same dtype, records the value in the feed mapping (mutated in place)
      # and returns the placeholder.
      if isinstance(args, tuple):
        return [placeholders(x, feed) for x in args]
      else:
        x = ops.convert_to_tensor(args).eval()
        fake = array_ops.placeholder(np.asarray(x).dtype)
        feed[fake] = x
        return fake

    # Test various GraphDef versions
    for version in strict + lenient:
      with ops.Graph().as_default() as g:
        test_util.set_producer_version(g, version)
        with self.session(graph=g) as sess:
          feed = {}
          xs = placeholders(args, feed)
          x = op(*xs)
          if version in strict:
            with self.assertRaisesOpError(error):
              sess.run(x, feed_dict=feed)
          else:
            r = sess.run(x, feed_dict=feed)
            if correct is not None:
              self.assertAllEqual(r, correct)

  def testConcat(self):
    # [FUNCTION DOC]
    # Checks that concat with a scalar element among its inputs raises the "cannot concatenate
    # scalars" error for strict producer versions, for three scalar/vector arrangements.
    for data in (2, [3], 7), ([2], 3, 7), ([2], [3], 7):
      self.check(array_ops.concat, (data, 0),
                 r"Can't concatenate scalars \(use tf.stack instead\)",
                 [2, 3, 7])

  def testFill(self):
    # [FUNCTION DOC]
    # Checks fill with a scalar dims argument and with a vector value argument; versions five and
    # six are lenient and expected to produce a result of threes, with no strict versions, so the
    # error pattern is never asserted.
    self.check(
        array_ops.fill, (2, 3),
        'dims must be a vector', [3, 3],
        lenient=[5, 6],
        strict=[])
    self.check(
        array_ops.fill, ([2], [3]),
        'value must be a scalar', [3, 3],
        lenient=[5, 6],
        strict=[])

  def testPad(self):
    # [FUNCTION DOC]
    # Checks pad on a scalar input with a one-row paddings matrix raises the error that the first
    # paddings dimension must equal the input rank, for strict versions.
    self.check(array_ops.pad, (7, [[1, 2]]),
               'The first dimension of paddings must be the rank of inputs',
               [0, 7, 0, 0])

  def testRandom(self):
    # [FUNCTION DOC]
    # Checks that random uniform with a scalar shape argument raises the "shape must be a vector"
    # error for strict versions.
    self.check(random_ops.random_uniform, (3,), 'shape must be a vector')

  def testReshape(self):
    # [FUNCTION DOC]
    # Checks reshape of a scalar with a one-element shape; versions five and six are lenient and
    # expected to give a one-element vector; there are no strict versions.
    self.check(
        array_ops.reshape, (7, 1),
        'sizes input must be 1-D', [7],
        lenient=[5, 6],
        strict=[])

  def testShardedFilename(self):
    # [FUNCTION DOC]
    # Checks the sharded filename op with a vector for the total shard count raises a "must be a
    # scalar" error for strict versions (expected legacy result: foo shard 4 of 100).
    self.check(gen_io_ops.sharded_filename, ('foo', 4, [100]),
               'must be a scalar', b'foo-00004-of-00100')

  def testShardedFilespec(self):
    # [FUNCTION DOC]
    # Checks the sharded filespec op with a vector shard count raises a "must be a scalar" error
    # for strict versions (expected legacy result: foo wildcard of 100).
    self.check(gen_io_ops.sharded_filespec, ('foo', [100]), 'must be a scalar',
               b'foo-?????-of-00100')

  def testUnsortedSegmentSum(self):
    # [FUNCTION DOC]
    # Checks unsorted segment sum with a vector number-of-segments argument raises an error that
    # it should be a scalar, for strict versions.
    self.check(math_ops.unsorted_segment_sum, (7, 1, [4]),
               'num_segments should be a scalar', [0, 7, 0, 0])

  def testRange(self):
    # [FUNCTION DOC]
    # Checks range with a vector start, limit or delta in turn; versions five and six are lenient
    # and expected to produce zero and two, with no strict versions.
    self.check(
        math_ops.range, ([0], 3, 2),
        'start must be a scalar', [0, 2],
        lenient=[5, 6],
        strict=[])
    self.check(
        math_ops.range, (0, [3], 2),
        'limit must be a scalar', [0, 2],
        lenient=[5, 6],
        strict=[])
    self.check(
        math_ops.range, (0, 3, [2]),
        'delta must be a scalar', [0, 2],
        lenient=[5, 6],
        strict=[])

  def testSlice(self):
    # [FUNCTION DOC]
    # Checks slice on ten numbers with scalar begin, scalar size, or both scalar/vector
    # combinations raises the error that begin and size must be 1-D tensors, for strict versions.
    data = np.arange(10)
    error = 'Expected begin and size arguments to be 1-D tensors'
    self.check(array_ops.slice, (data, 2, 3), error, [2, 3, 4])
    self.check(array_ops.slice, (data, [2], 3), error, [2, 3, 4])
    self.check(array_ops.slice, (data, 2, [3]), error, [2, 3, 4])

  def testSparseToDense(self):
    # [FUNCTION DOC]
    # Checks sparse-to-dense with a scalar output shape raises an error that the output shape must
    # be rank one, for strict versions.
    self.check(sparse_ops.sparse_to_dense, (1, 4, 7),
               'output_shape must be rank 1', [0, 7, 0, 0])

  def testTile(self):
    # [FUNCTION DOC]
    # Checks tile with a scalar multiples argument raises an error that multiples must be 1-D,
    # for strict versions.
    self.check(array_ops.tile, ([7], 2), 'Expected multiples to be 1-D', [7, 7])


if __name__ == '__main__':
  test.main()
