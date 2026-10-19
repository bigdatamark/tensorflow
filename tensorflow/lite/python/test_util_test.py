# [MODULE DOC]
# Unit tests for the test_util helper module in this package. Currently covers the function that
# lists operator names in a TFLite model, using small prebuilt model files from the testdata
# directory. Runs via the standard TensorFlow test runner when executed as a script.
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
"""Tests for test_util.py."""

from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class TestUtilTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the ops-listing helper of the tflite test_util module. Has no shared
  # fixtures; each test loads a prebuilt model file from the testdata directory.

  def testBuiltinOp(self):
    # [FUNCTION DOC]
    # Scenario: read the small "add" model file from testdata as bytes. Action: list its ops.
    # Asserts the result contains exactly one op name, the builtin name for addition.
    model_path = resource_loader.get_path_to_datafile('../testdata/add.bin')
    op_set = tflite_test_util.get_ops_list(gfile.GFile(model_path, 'rb').read())
    self.assertCountEqual(op_set, ['ADD'])

  def testFlexOp(self):
    # [FUNCTION DOC]
    # Scenario: read the "softplus_flex" model file from testdata, which uses a Flex (select
    # TensorFlow) op. Action: list its ops. Asserts the result contains exactly one name, the
    # custom op name for the Flex softplus op.
    model_path = resource_loader.get_path_to_datafile(
        '../testdata/softplus_flex.bin')
    op_set = tflite_test_util.get_ops_list(gfile.GFile(model_path, 'rb').read())
    self.assertCountEqual(op_set, ['FlexSoftplus'])


if __name__ == '__main__':
  test.main()
