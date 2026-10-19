# [MODULE DOC]
# XLA test checking that the gradient of image resizing can be compiled with just-in-time
# compilation and gives the same result as the uncompiled run. The module docstring refers to
# a repeat op, which does not match the content. Enables eager execution when run as a program.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.array_ops.repeat."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ImageOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for image op gradients under jit compilation. Stateless.

  def testGradImageResize(self):
    # [FUNCTION DOC]
    # On the test device, builds a scalar variable, a model that scales its input by the
    # variable and bilinearly resizes it to 2048 by 2048, and a train function returning the
    # gradient of the mean squared error with respect to the variable. Runs it eagerly and with
    # jit compilation on zero inputs and asserts the two gradients are close.
    """Tests that the gradient of image.resize is compilable."""
    with ops.device("device:{}:0".format(self.device)):
      img_width = 2048
      var = variables.Variable(array_ops.ones(1, dtype=dtypes.float32))

      def model(x):
        # [FUNCTION DOC]
        # Multiplies the input by the captured variable and bilinearly resizes the result to the
        # square image size; returns the resized tensor.
        x = var * x
        x = image_ops.resize_images(
            x,
            size=[img_width, img_width],
            method=image_ops.ResizeMethod.BILINEAR)
        return x

      def train(x, y):
        # [FUNCTION DOC]
        # Inputs: a low-resolution image batch and a target batch. Runs the model under a
        # gradient recorder, computes the mean squared difference to the target, and returns the
        # list of gradients with respect to the captured variable.
        with backprop.GradientTape() as tape:
          output = model(x)
          loss_value = math_ops.reduce_mean((y - output)**2)
        grads = tape.gradient(loss_value, [var])
        return grads

      compiled_train = def_function.function(train, jit_compile=True)
      x = array_ops.zeros((1, img_width // 2, img_width // 2, 1),
                          dtype=dtypes.float32)
      y = array_ops.zeros((1, img_width, img_width, 1), dtype=dtypes.float32)
      self.assertAllClose(train(x, y), compiled_train(x, y))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
