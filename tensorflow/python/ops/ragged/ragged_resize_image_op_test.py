# [MODULE DOC]
# Unit tests for dispatching the image resize operations (both the version-one and version-two
# functions) to ragged batches of images with varying height and width. Checks the dense result
# shape and that each resized image matches resizing that image on its own, including partially
# known static shapes, tensor-valued sizes and rank validation. Runs test main as a script.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for RaggedTensor dispatch of tf.images.resize."""

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedResizeImageOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Test case for resizing ragged image batches, run in graph and eager modes. Provides a
  # helper method that builds ragged image batches from a list of sizes.

  def make_image_batch(self, sizes, channels):
    # [FUNCTION DOC]
    # HELPER: Fixture builder for the resize tests.
    # Inputs: sizes (list of height and width pairs, possibly empty) and channels (number of
    # channels). Returns a ragged tensor of images: for an empty list, a zero-image batch of
    # five by five images with ragged rank two; otherwise each image is filled with increasing
    # float values and the images are stacked into a ragged batch.
    if not sizes:
      return ragged_tensor.RaggedTensor.from_tensor(
          array_ops.zeros([0, 5, 5, channels]), ragged_rank=2)
    images = [
        array_ops.reshape(
            math_ops.range(w * h * channels * 1.0), [w, h, channels])
        for (w, h) in sizes
    ]
    return ragged_concat_ops.stack(images)

  @parameterized.parameters([
      dict(src_sizes=[], dst_size=(4, 4), v1=True),
      dict(src_sizes=[], dst_size=(4, 4), v1=False),
      dict(src_sizes=[(2, 2)], dst_size=(4, 4), v1=True),
      dict(src_sizes=[(2, 2)], dst_size=(4, 4), v1=False),
      dict(src_sizes=[(2, 8), (3, 5), (10, 10)], dst_size=(5, 5), v1=True),
      dict(src_sizes=[(2, 8), (3, 5), (10, 10)], dst_size=(5, 5), v1=False),
  ])
  def testResize(self, src_sizes, dst_size, v1=False):
    # [FUNCTION DOC]
    # Parameters: src_sizes (image sizes in the batch), dst_size (target height and width) and
    # v1 (use the version-one resize function instead of version two). Resizes a ragged batch of
    # three-channel images, asserts the result is a regular tensor with the expected static
    # shape, and asserts each resized image is close to resizing that image alone.
    resize = image_ops.resize_images if v1 else image_ops.resize_images_v2

    # Construct the input images.
    channels = 3
    images = self.make_image_batch(src_sizes, channels)
    expected_shape = [len(src_sizes)] + list(dst_size) + [channels]

    # Resize the ragged batch of images.
    resized_images = resize(images, dst_size)
    self.assertIsInstance(resized_images, tensor.Tensor)
    self.assertEqual(resized_images.shape.as_list(), expected_shape)

    # Check that results for each image matches what we'd get with the
    # non-batch version of tf.images.resize.
    for i in range(len(src_sizes)):
      actual = resized_images[i]
      expected = resize(images[i].to_tensor(), dst_size)
      self.assertAllClose(actual, expected)

  @parameterized.parameters([
      dict(src_shape=[None, None, None, None], src_sizes=[], dst_size=(4, 4)),
      dict(src_shape=[None, None, None, 3], src_sizes=[], dst_size=(4, 4)),
      dict(src_shape=[0, None, None, None], src_sizes=[], dst_size=(4, 4)),
      dict(src_shape=[0, None, None, 3], src_sizes=[], dst_size=(4, 4)),
      dict(
          src_shape=[None, None, None, None],
          src_sizes=[(2, 2)],
          dst_size=(4, 4)),
      dict(
          src_shape=[None, None, None, None],
          src_sizes=[(2, 8), (3, 5), (10, 10)],
          dst_size=(5, 5)),
      dict(
          src_shape=[None, None, None, 1],
          src_sizes=[(2, 8), (3, 5), (10, 10)],
          dst_size=(5, 5)),
      dict(
          src_shape=[3, None, None, 1],
          src_sizes=[(2, 8), (3, 5), (10, 10)],
          dst_size=(5, 5)),
  ])
  def testResizeWithPartialStaticShape(self, src_shape, src_sizes, dst_size):
    # [FUNCTION DOC]
    # Parameters: src_shape (partially known static shape of the input), src_sizes and dst_size.
    # Builds a ragged batch, runs the version-two resize inside a compiled function whose input
    # signature uses the partial shape, asserts the output is a regular tensor with a shape
    # compatible with the expected one, and each resized image is close to resizing it alone.
    channels = src_shape[-1] or 3
    images = self.make_image_batch(src_sizes, channels)
    rt_spec = ragged_tensor.RaggedTensorSpec(src_shape,
                                             ragged_rank=images.ragged_rank)
    expected_shape = [len(src_sizes)] + list(dst_size) + [channels]

    # Use @tf.function to erase static shape information.
    @def_function.function(input_signature=[rt_spec])
    def do_resize(images):
      # [FUNCTION DOC]
      # Nested function under test: resizes the ragged image batch to the enclosing target size
      # using the version-two resize function and returns the result.
      return image_ops.resize_images_v2(images, dst_size)

    resized_images = do_resize(images)
    self.assertIsInstance(resized_images, tensor.Tensor)
    self.assertTrue(resized_images.shape.is_compatible_with(expected_shape))

    # Check that results for each image matches what we'd get with the
    # non-batch version of tf.images.resize.
    for i in range(len(src_sizes)):
      actual = resized_images[i]
      expected = image_ops.resize_images_v2(images[i].to_tensor(), dst_size)
      self.assertAllClose(actual, expected)

  def testSizeIsTensor(self):
    # [FUNCTION DOC]
    # Calls a compiled resize function with the target size given as a tensor, on a ragged batch
    # of three images, and asserts the output is a regular tensor compatible with the expected
    # shape of three images, two by two, three channels.
    @def_function.function
    def do_resize(images, new_size):
      # [FUNCTION DOC]
      # Nested function under test: resizes the images to the given size tensor and returns it.
      return image_ops.resize_images_v2(images, new_size)

    src_images = self.make_image_batch([[5, 8], [3, 2], [10, 4]], 3)
    resized_images = do_resize(src_images, constant_op.constant([2, 2]))
    self.assertIsInstance(resized_images, tensor.Tensor)
    self.assertTrue(resized_images.shape.is_compatible_with([3, 2, 2, 3]))

  def testBadRank(self):
    # [FUNCTION DOC]
    # Asserts that resizing a ragged tensor converted from a rank-three dense tensor raises a
    # value error stating the rank must be four.
    rt = ragged_tensor.RaggedTensor.from_tensor(array_ops.zeros([5, 5, 3]))
    with self.assertRaisesRegex(ValueError, 'rank must be 4'):
      image_ops.resize_images_v2(rt, [10, 10])


if __name__ == '__main__':
  googletest.main()
