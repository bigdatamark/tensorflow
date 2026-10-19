# [MODULE DOC]
# Test module checking that requesting XLA compilation on a build without XLA support fails with
# a clear unimplemented error. Meant to run only without the CUDA configuration (which links in
# the XLA JIT). Defines one test case class; running as a script enables eager execution and
# starts the test runner. No import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class FunctionCpuOnlyTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for CPU-only builds (no XLA JIT linked in). Combines the framework test base class
  # with parameterized testing; it has no shared fixtures and holds a single test verifying
  # that requesting XLA compilation fails cleanly.
  """Test that jit_compile=True correctly throws an exception if XLA is not available.

  This test should only be run without `--config=cuda`, as that implicitly links
  in XLA JIT.
  """

  def testJitCompileRaisesExceptionWhenXlaIsUnsupported(self):
    # [FUNCTION DOC]
    # Scenario: a traced function is declared with XLA compilation enabled and then called with
    # a small list of integers. Takes no inputs and returns nothing. If the build uses ROCm or
    # XLA is enabled for tests, the test returns immediately without checking anything.
    # Otherwise it asserts that calling the function raises an unimplemented error whose message
    # says support for that platform is not linked in.
    if test.is_built_with_rocm() or test_util.is_xla_enabled():
      return

    with self.assertRaisesRegex(errors.UnimplementedError,
                                'support for that platform linked in'):

      @polymorphic_function.function(jit_compile=True)
      def fn(x):
        # [FUNCTION DOC]
        # Function compiled with XLA requested; returns its input added to itself. Calling it is
        # expected to fail on builds without XLA support.
        return x + x

      fn([1, 1, 2, 3])


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
