# [MODULE DOC]
# Test module for the checkpoint Saver with very large partitioned variables. It defines one
# test case that saves a partitioned boolean variable whose combined size is larger than the
# 32-bit integer range, to verify that partitioning lets the save succeed. Kept separate from
# the main saver tests because of memory demands. Has no import-time side effects; running it
# as a script starts the test runner.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver


class SaverLargePartitionedVariableTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Saver with a partitioned variable whose total size exceeds the 32-bit
  # integer range. It is kept in its own file because of the large memory the test needs. It has
  # no shared fixtures beyond the temporary directory provided by the base test case.

  # Need to do this in a separate test because of the amount of memory needed
  # to run this test.
  def testLargePartitionedVariables(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph and session, on the first CPU device, create a boolean variable
    # with 2^31 elements split by a fixed-size partitioner into four partitions, each small
    # enough to be serialized. Every partition is initialized to true. After running the global
    # variable initializer, a Saver built over the list of partitions saves the checkpoint to a
    # file in the test temporary directory. Expected: save returns exactly the path that was
    # requested. Side effects: writes a large checkpoint file; needs a lot of memory.
    save_path = os.path.join(self.get_temp_dir(), "large_variable")
    var_name = "my_var"
    # Saving large partition variable.
    with session.Session("", graph=ops.Graph()) as sess:
      with ops.device("/cpu:0"):
        # Create a partitioned variable which is larger than int32 size but
        # split into smaller sized variables.
        init = lambda shape, dtype, partition_info: constant_op.constant(
            True, dtype, shape)
        partitioned_var = list(variable_scope.get_variable(
            var_name,
            shape=[1 << 31],
            partitioner=partitioned_variables.fixed_size_partitioner(4),
            initializer=init,
            dtype=dtypes.bool))
        self.evaluate(variables.global_variables_initializer())
        save = saver.Saver(partitioned_var)
        val = save.save(sess, save_path)
        self.assertEqual(save_path, val)


if __name__ == "__main__":
  test.main()
