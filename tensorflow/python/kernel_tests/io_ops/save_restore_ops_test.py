# [MODULE DOC]
# Test module for the checkpoint save/restore and sharded file name ops in io_ops. Covers saving
# and restoring a scalar and a sliced tensor through relative checkpoint paths for several element
# types (float32, int32 and the 4-bit integer types), the sharded file name and file spec ops,
# and static shape inference (and validation errors) of the restore ops in graph construction.
# Defines the module-level list of element types used to parameterize the tests. Running the
# file as a script starts the TensorFlow test runner. No other import-time side effects.
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
"""Tests for tensorflow.ops.io_ops."""
import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test

_TEST_DTYPES = [dtypes.float32, dtypes.int32, dtypes.int4, dtypes.uint4]


class SaveRestoreTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests (one case per element type) of the version 2 save and restore ops writing
  # and reading a checkpoint prefix relative to the current working directory, which each test
  # changes to the test temp directory.

  @parameterized.parameters(_TEST_DTYPES)
  @test_util.run_in_graph_and_eager_modes
  def testRelativePath(self, dtype):
    # [FUNCTION DOC]
    # Scenario: for the given element type (an argument supplied by the parameterization), changes
    # the working directory to the temp directory (a process-wide side effect), saves a scalar
    # constant two under the name x with a relative checkpoint prefix, then restores it. Asserts
    # the restored value equals two. Runs in both graph and eager modes.
    os.chdir(self.get_temp_dir())
    self.evaluate(
        io_ops.save_v2(
            "ckpt", ["x"], [""], [constant_op.constant(2, dtype=dtype)]
        )
    )
    self.assertAllEqual(
        [2], self.evaluate(io_ops.restore_v2("ckpt", ["x"], [""], [dtype]))
    )

  @parameterized.parameters(_TEST_DTYPES)
  def testWithSliceInput(self, dtype):
    # [FUNCTION DOC]
    # Scenario: for the given element type, changes the working directory to the temp directory,
    # saves a two by three matrix under the name x, then restores only a slice (a slice spec
    # taking all rows and just the second column). Asserts the restored slice equals a column
    # holding the values two and three.
    # Side effect: changes the process working directory.
    os.chdir(self.get_temp_dir())
    self.evaluate(
        io_ops.save_v2(
            "ckpt",
            ["x"],
            [""],
            [constant_op.constant([[1, 2, 3], [2, 3, 4]], dtype=dtype)],
        )
    )
    self.assertAllEqual(
        [[2], [3]],
        self.evaluate(io_ops.restore_v2("ckpt", ["x"], ["2 3 -:1,1"], [dtype]))[
            0
        ],
    )


class ShardedFileOpsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the ops that build sharded checkpoint file names and file specs.

  def testShardedFileName(self):
    # [FUNCTION DOC]
    # Scenario: inside a session configured with two CPU devices, evaluates the sharded file name
    # op for base foo, shard four of one hundred, and the sharded file spec op for base foo with
    # one hundred shards. Asserts the name is the base, a five digit shard number, "of" and a five
    # digit shard count, and that the spec uses question marks in place of the shard number.
    with session.Session(
        target="", config=config_pb2.ConfigProto(device_count={"CPU": 2})):
      self.assertEqual(
          gen_io_ops.sharded_filename("foo", 4, 100).eval(),
          b"foo-00004-of-00100")
      self.assertEqual(
          gen_io_ops.sharded_filespec("foo", 100).eval(), b"foo-?????-of-00100")


class ShapeInferenceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized (per element type) tests of static shape inference and argument validation of
  # the restore ops, performed purely at graph construction time in a fresh graph.

  @parameterized.parameters(_TEST_DTYPES)
  def testRestoreV2WithSliceInput(self, dtype):
    # [FUNCTION DOC]
    # Scenario: builds a version 2 restore op for two tensors, one with no slice (unknown shape)
    # and one with a slice spec for a three by four tensor selecting the first row only.
    # Asserts the op has two outputs, the first output's shape is not fully defined, and the
    # second has static shape one by four.
    with ops.Graph().as_default():
      op = io_ops.restore_v2(
          "model", ["var1", "var2"], ["", "3 4 0,1:-"], [dtype, dtype]
      )
      self.assertEqual(2, len(op))
      self.assertFalse(op[0].get_shape().is_fully_defined())
      self.assertEqual([1, 4], op[1].get_shape())

  @parameterized.parameters(_TEST_DTYPES)
  def testRestoreV2NumSlicesNotMatch(self, dtype):
    # [FUNCTION DOC]
    # Scenario: builds a version 2 restore op with three tensor names but only two slice specs.
    # Asserts graph construction raises a value error because the counts do not match.
    with ops.Graph().as_default():
      with self.assertRaises(ValueError):
        io_ops.restore_v2(
            "model", ["var1", "var2", "var3"], ["", "3 4 0,1:-"], [dtype, dtype]
        )

  @parameterized.parameters(_TEST_DTYPES)
  def testRestoreSlice(self, dtype):
    # [FUNCTION DOC]
    # Scenario: builds the older restore-slice op with a slice spec over a three by four shape
    # taking one row. Asserts the output static shape is one by four.
    with ops.Graph().as_default():
      op = gen_io_ops.restore_slice("model", "var", "3 4 0,1:-", dtype)
      self.assertEqual([1, 4], op.get_shape())


if __name__ == "__main__":
  test.main()
