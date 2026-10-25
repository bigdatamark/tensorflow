# [MODULE DOC]
# Concrete test module for the encode_proto op. Subclasses the shared encode test base and binds both
# the encode and decode modules to the public proto ops module, so the table-driven encode tests run
# against the standard ops. Running the file as a script starts the TensorFlow test runner.
# =============================================================================
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
# =============================================================================
"""Tests for encode_proto op."""

# Python3 readiness boilerplate
from tensorflow.python.kernel_tests.proto import encode_proto_op_test_base as test_base
from tensorflow.python.ops import proto_ops
from tensorflow.python.platform import test


class EncodeProtoOpTest(test_base.EncodeProtoOpTestBase):
  # [CLASS DOC]
  # Test case running the shared encode_proto tests with the real proto ops module used for both
  # encoding and decoding. Adds no tests of its own.

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor. Takes the test method name (default runTest) and calls the base constructor with
    # the proto ops module as both the encode and decode module. Returns nothing.
    super(EncodeProtoOpTest, self).__init__(encode_module=proto_ops,
                                            decode_module=proto_ops,
                                            methodName=methodName)


if __name__ == '__main__':
  test.main()
