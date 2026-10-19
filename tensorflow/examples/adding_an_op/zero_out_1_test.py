# [MODULE DOC]
# Tests for version 1 of the zero-out example op wrapper module. Contains one test case class
# covering the plain op, namespaced variants and repeated library loading, plus a standard main
# entry running the test runner.
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

"""Test for version 1 of the zero_out op."""

import os.path

import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_op_1


class ZeroOut1Test(tf.test.TestCase):
  # [CLASS DOC]
  # Test case covering version 1 of the zero-out example op. Has no fixtures.

  def test(self):
    # [FUNCTION DOC]
    # Applies the zero-out op to five, four, three, two, one and asserts only the first element
    # is kept: five followed by four zeros.
    result = zero_out_op_1.zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_namespace(self):
    # [FUNCTION DOC]
    # Same scenario as the basic test but using the op registered inside a namespace; expects
    # five followed by four zeros.
    result = zero_out_op_1.namespace_zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_namespace_call_op_on_op(self):
    # [FUNCTION DOC]
    # Feeds the output of the namespaced zero-out op into the same op again and asserts the
    # result is still five followed by four zeros (the op is idempotent here).
    x = zero_out_op_1.namespace_zero_out([5, 4, 3, 2, 1])
    result = zero_out_op_1.namespace_zero_out(x)
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_namespace_nested(self):
    # [FUNCTION DOC]
    # Applies the zero-out op registered in a nested namespace to five, four, three, two, one and
    # asserts five followed by four zeros.
    result = zero_out_op_1.namespace_nested_zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_load_twice(self):
    # [FUNCTION DOC]
    # Loads the compiled zero-out kernel library a second time from the data files directory and
    # asserts the returned module is the very same object as the one already loaded by the
    # wrapper module (loading is cached).
    zero_out_loaded_again = tf.load_op_library(os.path.join(
        tf.compat.v1.resource_loader.get_data_files_path(),
        'zero_out_op_kernel_1.so'))
    self.assertEqual(zero_out_loaded_again, zero_out_op_1._zero_out_module)


if __name__ == '__main__':
  tf.test.main()
