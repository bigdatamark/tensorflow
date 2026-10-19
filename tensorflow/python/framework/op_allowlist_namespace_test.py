# [MODULE DOC]
# Test module checking that the Python wrapper generated for an op with a namespace, built via the
# op allowlist mechanism, exists and can be imported. Runs under the googletest runner.
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
"""# Test that building using op_allowlist works with ops with namespaces."""

from tensorflow.python.framework import test_namespace_ops
from tensorflow.python.platform import googletest


class OpAllowlistNamespaceTest(googletest.TestCase):
  # [CLASS DOC]
  # Test case covering the generated wrapper module for namespaced ops built with an op allowlist.

  def testOpAllowListNamespace(self):
    # [FUNCTION DOC]
    # Test: looks up the namespaced string-output test op in the generated test namespace ops
    # module and asserts it is not null, proving the wrapper was generated. No side effects.
    """Test that the building of the python wrapper worked."""
    op = test_namespace_ops.namespace_test_string_output
    self.assertIsNotNone(op)


if __name__ == "__main__":
  googletest.main()
