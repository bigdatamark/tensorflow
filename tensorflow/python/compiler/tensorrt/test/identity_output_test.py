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
# [MODULE DOC]
# Regression test for TF-TensorRT conversion where one tensor is exposed as several graph outputs
# through duplicated identity ops (the output binding name used to be overwritten, causing a
# runtime error). Built on TfTrtIntegrationTestBase; runs via the standard test main.
"""This test checks a situation where the same tensor is considered as an output

multiple times because it has been duplicated by 2+ identity ops. Previously,
the tensor would be renamed multiple times, overwriting the output binding name
which resulted in a runtime error when the binding would not be found.
"""

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class IdentityTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that an engine whose single result tensor is returned three times through identity ops
  # (one chained through an extra identity) builds and runs correctly.
  """Testing engine with the same tensor repeated as output via identity."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph: exponential of the input added to the input, then three identity outputs
    # named output_0, output_1 and output_2 (the last through an intermediate unnamed identity).
    # Returns the list of the three outputs.
    x1 = math_ops.exp(x)
    x1 = x1 + x

    out1 = array_ops.identity(x1, name='output_0')
    out2 = array_ops.identity(x1, name='output_1')
    iden1 = array_ops.identity(x1)
    out3 = array_ops.identity(iden1, name='output_2')
    return [out1, out2, out3]

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, one input of shape 100x32 and three outputs each of shape 100x32.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 32]],
                            [[100, 32]] * 3)

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    """Return the expected engines to build."""
    return ['TRTEngineOp_000']


if __name__ == '__main__':
  test.main()
