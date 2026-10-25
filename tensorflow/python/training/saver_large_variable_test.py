# [MODULE DOC]
# Test module for the checkpoint Saver with a single huge variable. It defines one test case
# checking that saving a 2 GB unpartitioned variable in the version 1 format fails with an
# invalid-argument error. Separate from the main saver tests because large allocations are
# flaky under the thread sanitizer. No import-time side effects; running it as a script starts
# the test runner.
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
# =============================================================================
"""Tests for tensorflow.python.training.saver.py."""

import os

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver


class SaverLargeVariableTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Saver's behavior when a single variable is too large to serialize.
  # Kept in a separate file from the main saver tests because huge allocations are flaky under
  # the thread sanitizer. It has no shared fixtures beyond the base test case's temp directory.

  # NOTE: This is in a separate file from saver_test.py because the
  # large allocations do not play well with TSAN, and cause flaky
  # failures.
  def testLargeVariable(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph and session, on the first CPU device, create an unpartitioned
    # boolean variable of exactly 2 GB (shape 2 x 1024 x 1024 x 1024). A Saver is built over a
    # mapping from the variable's op name to the variable, using the legacy version 1 checkpoint
    # format. After running the variable initializer, saving is attempted. Expected: save raises
    # an invalid-argument error whose message says the tensor slice is too large to serialize,
    # since the checkpoint adds header metadata on top of the 2 GB payload.
    save_path = os.path.join(self.get_temp_dir(), "large_variable")
    with session.Session("", graph=ops.Graph()) as sess:
      # Declare a variable that is exactly 2GB. This should fail,
      # because a serialized checkpoint includes other header
      # metadata.
      with ops.device("/cpu:0"):
        var = variables.Variable(
            constant_op.constant(
                False, shape=[2, 1024, 1024, 1024], dtype=dtypes.bool))
      save = saver.Saver(
          {
              var.op.name: var
          }, write_version=saver_pb2.SaverDef.V1)
      var.initializer.run()
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Tensor slice is too large to serialize"):
        save.save(sess, save_path)


if __name__ == "__main__":
  test.main()
