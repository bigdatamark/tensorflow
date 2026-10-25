# [MODULE DOC]
# Tests for array operations when inputs are weak tensors (tensors whose dtype is inferred and may
# be promoted). Importing the weak tensor ops module registers the weak-tensor-aware op overrides.
# Covers shape inference through reshape and through compiled functions. When run as a script it
# first enables the "all" dtype conversion mode and then starts the test runner.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for array operations."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.platform import test


class ArrayOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering array operations (reshape, zeros, identity) applied to weak tensors and
  # shape inference in compiled functions. Holds no shared fixtures; each test builds its own
  # inputs.

  def testReshapeShapeInference(self):
    # [FUNCTION DOC]
    # Builds a weak tensor from random normal data of shape 4 by 10 by 10 and reshapes it to its
    # own dynamic shape, both as the default integer type and as 64-bit integers. Asserts the
    # static shape stays compatible with 4, unknown, 10. Then reshapes using a shape that went
    # through a cast to float and back to 32-bit integer; shape inference must not see through
    # that cast, so all dimensions are unknown, and the result must still be a weak tensor.
    # Create a tensor with an unknown dim 1.
    x = weak_tensor.WeakTensor(random_ops.random_normal([4, 10, 10]))
    x.shape.assert_is_compatible_with([4, None, 10])
    a = array_ops.reshape(x, array_ops.shape(x))
    a.shape.assert_is_compatible_with([4, None, 10])
    b = array_ops.reshape(x, math_ops.cast(array_ops.shape(x), dtypes.int64))
    b.shape.assert_is_compatible_with([4, None, 10])

    # We do not shape-infer across a tf.cast into anything that's not tf.int32
    # or tf.int64, since they might end up mangling the shape.
    c = array_ops.reshape(
        x,
        math_ops.cast(
            math_ops.cast(array_ops.shape(x), dtypes.float32), dtypes.int32
        ),
    )
    c.shape.assert_is_compatible_with([None, None, None])
    self.assertIsInstance(c, weak_tensor.WeakTensor)

  def testSlicedPartialShapeInference(self):
    # [FUNCTION DOC]
    # Traces a compiled function (autograph off) that creates a zeros tensor whose size is the
    # first dimension of the dynamic shape of its input, using an input spec of 10 by unknown.
    # Asserts the concrete function's output shape is known to be a single dimension of size 10.
    @def_function.function(autograph=False)
    def g(x):
      # [FUNCTION DOC]
      # Takes a tensor and returns a zeros vector whose length equals the tensor's first
      # dimension read from its dynamic shape.
      return array_ops.zeros([array_ops.shape(x)[0]])

    conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
    self.assertAllEqual(conc.output_shapes.as_list(), [10])

  def testIdentityOnSlicedPartialShapeInference(self):
    # [FUNCTION DOC]
    # Same scenario as the sliced partial shape test, but the first dimension is passed through an
    # identity operation before being used as the zeros size. Asserts the concrete function's
    # output shape is still inferred as a single dimension of size 10.
    @def_function.function(autograph=False)
    def g(x):
      # [FUNCTION DOC]
      # Takes a tensor and returns a zeros vector whose length is the identity of the tensor's
      # first dimension read from its dynamic shape.
      return array_ops.zeros([array_ops.identity(array_ops.shape(x)[0])])

    conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
    self.assertAllEqual(conc.output_shapes.as_list(), [10])


if __name__ == "__main__":
  ops.set_dtype_conversion_mode("all")
  test.main()
