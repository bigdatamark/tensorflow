# [MODULE DOC]
# Concrete test module for the decode_proto op exposed through the proto_ops Python module.
# It only binds the shared test-case base class (decode_proto_op_test_base) to proto_ops, so every
# test defined in the base class runs against the decode_proto wrappers in proto_ops. Running the
# file as a script starts the TensorFlow test runner. No registrations or flags at import time.
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
"""Tests for decode_proto op."""

# Python3 preparedness imports.
from tensorflow.python.kernel_tests.proto import decode_proto_op_test_base as test_base
from tensorflow.python.ops import proto_ops as proto_ops
from tensorflow.python.platform import test


class DecodeProtoOpTest(test_base.DecodeProtoOpTestBase):
  # [CLASS DOC]
  # Test case that runs all decode_proto tests inherited from the shared base class, using the
  # proto_ops module as the provider of the decode functions. Holds no state of its own; the
  # module under test is passed to the base constructor.

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor. Takes the name of the test method to run (optional, defaults to the standard
    # run-test name) and forwards it to the base class together with the proto_ops module, which
    # the base class uses as the decode/encode implementation. Returns nothing; no side effects.
    super(DecodeProtoOpTest, self).__init__(proto_ops, methodName)


if __name__ == '__main__':
  test.main()
