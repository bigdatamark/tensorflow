# [MODULE DOC]
# Reference test for AutoGraph handling of Python type annotations. Defines a subject function
# containing a bare variable annotation without assignment, and a test class that checks the
# compiled function matches eager execution. No import-time side effects; runs the TF test main
# when executed directly.
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
"""Type annotations."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def pure_declaration():
  # [FUNCTION DOC]
  # Test subject: contains an annotation-only declaration of an integer variable (no value is
  # assigned) and then returns the constant 1. Takes no inputs.
  n: int  # pylint:disable=unused-variable
  return 1


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Holds the single reference test for type-annotation handling.

  def test_pure_declaration(self):
    # [FUNCTION DOC]
    # Asserts the compiled annotation-only subject function returns the same result as eager.
    self.assertFunctionMatchesEager(pure_declaration)


if __name__ == '__main__':
  tf.test.main()
