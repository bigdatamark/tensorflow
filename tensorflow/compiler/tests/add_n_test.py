# [MODULE DOC]
# Test module for the add-n operation applied to tensor lists on an XLA test device. It checks
# successful summation of lists and the invalid-argument errors raised on shape mismatch. It
# exposes no public API; running it as a script starts the test runner. No import-time effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for AddN."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class XlaAddNTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the add-n operation when its inputs are tensor lists (variant values),
  # covering the success path and two shape-mismatch failure paths. Holds no state of its own.

  def testAddTensorLists(self):
    # [FUNCTION DOC]
    # Creates two reserved float tensor lists of three scalar elements, sets element 0 of the
    # first to 5 and element 2 of the second to 10, adds them with add-n and stacks the result.
    # Asserts the stacked values are 5, 0 and 10 (unset elements count as zero).
    with self.session(), self.test_scope():
      l1 = list_ops.tensor_list_reserve(
          element_shape=[], element_dtype=dtypes.float32, num_elements=3)
      l2 = list_ops.tensor_list_reserve(
          element_shape=[], element_dtype=dtypes.float32, num_elements=3)
      l1 = list_ops.tensor_list_set_item(l1, 0, 5.)
      l2 = list_ops.tensor_list_set_item(l2, 2, 10.)

      l = math_ops.add_n([l1, l2])
      self.assertAllEqual(
          list_ops.tensor_list_stack(l, element_dtype=dtypes.float32),
          [5.0, 0.0, 10.0])

  def testAddTensorListsFailsIfLeadingDimsMismatch(self):
    # [FUNCTION DOC]
    # Reserves two tensor lists with different lengths (2 and 3), adds them with add-n and
    # evaluates the stacked result. Asserts an invalid-argument error whose message says list
    # arguments to add-n must all have the same shape.
    with self.session(), self.test_scope():
      l1 = list_ops.tensor_list_reserve(
          element_shape=[], element_dtype=dtypes.float32, num_elements=2)
      l2 = list_ops.tensor_list_reserve(
          element_shape=[], element_dtype=dtypes.float32, num_elements=3)
      l = math_ops.add_n([l1, l2])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "TensorList arguments to AddN must all have the same shape"):
        list_ops.tensor_list_stack(l, element_dtype=dtypes.float32).eval()

  def testAddTensorListsFailsIfElementShapesMismatch(self):
    # [FUNCTION DOC]
    # Builds two equal-length tensor lists whose element shapes come from placeholders, so static
    # shape inference cannot detect the problem. Runs the stacked sum in the session feeding an
    # empty (scalar) shape for one list and a two-element shape for the other. Asserts an
    # invalid-argument error saying list arguments to add-n must all have the same shape.
    with self.session() as session, self.test_scope():
      # Use placeholders instead of constant values for shapes to prevent TF's
      # shape inference from catching this early.
      l1_element_shape = array_ops.placeholder(dtype=dtypes.int32)
      l2_element_shape = array_ops.placeholder(dtype=dtypes.int32)
      l1 = list_ops.tensor_list_reserve(
          element_shape=l1_element_shape,
          element_dtype=dtypes.float32,
          num_elements=3)
      l2 = list_ops.tensor_list_reserve(
          element_shape=l2_element_shape,
          element_dtype=dtypes.float32,
          num_elements=3)
      l = math_ops.add_n([l1, l2])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "TensorList arguments to AddN must all have the same shape"):
        session.run(
            list_ops.tensor_list_stack(l, element_dtype=dtypes.float32), {
                l1_element_shape: [],
                l2_element_shape: [2]
            })


if __name__ == "__main__":
  test.main()
