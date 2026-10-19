# [MODULE DOC]
# GPU-specific test for XLA devices: verifies tensors can be copied between a physical GPU device
# and an XLA CPU device within one graph. Enables XLA devices in the eager context when the test
# case is constructed. Runs through the standard test main entry point.
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
"""Test cases for XLA devices."""

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class XlaDeviceGpuTest(test.TestCase):
  # [CLASS DOC]
  # Plain TensorFlow test case (not the XLA test base) that exercises GPU and XLA device
  # interaction. Its constructor turns on XLA devices globally for the eager context.

  def __init__(self, method_name="runTest"):
    # [FUNCTION DOC]
    # Constructor. Takes the test method name (default is the generic run method), forwards it to
    # the base class, then enables XLA devices in the global eager context (a process-wide side
    # effect). Returns nothing.
    super(XlaDeviceGpuTest, self).__init__(method_name)
    context.context().enable_xla_devices()

  def testCopiesToAndFromGpuWork(self):
    # [FUNCTION DOC]
    # Returns immediately (passes trivially) if no physical GPU exists. Otherwise builds a graph
    # that doubles a placeholder on the GPU, squares it on the XLA CPU device, then adds the two
    # results on the GPU. Feeds 1.5 and 0.5 and asserts the outputs are 12 and 2, proving copies in
    # both directions work.
    """Tests that copies between GPU and XLA devices work."""
    if not config.list_physical_devices("GPU"):
      return

    with session_lib.Session() as sess:
      x = array_ops.placeholder(dtypes.float32, [2])
      with ops.device("GPU"):
        y = x * 2
      with ops.device("device:XLA_CPU:0"):
        z = y * y
      with ops.device("GPU"):
        w = y + z
      result = sess.run(w, {x: [1.5, 0.5]})
    self.assertAllClose(result, [12., 2.], rtol=1e-3)


if __name__ == "__main__":
  test.main()
