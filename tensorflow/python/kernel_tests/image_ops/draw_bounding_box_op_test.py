# [MODULE DOC]
# Unit tests for the draw-bounding-boxes image op and, on GPU only, the generate-bounding-box-
# proposals op. The draw tests compare the op output against a reference image whose border is
# painted with the colour the op is expected to choose, checking that the colour table cycles
# once the number of boxes exceeds the number of colours, for default and user-supplied colour
# tables, for 1, 3 and 4 channel images and for float32 and half precision. No import-time side
# effects besides the test main entry point when run as a script.
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
"""Tests for draw_bounding_box_op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class DrawBoundingBoxOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the draw-bounding-boxes op. Two private helpers build the expected image
  # (border fill) and run the colour-cycling comparison; the public tests choose image shape,
  # colour table and dtype. A separate GPU-only test checks the error for bad proposal inputs.

  def _fillBorder(self, image, color):
    # [FUNCTION DOC]
    # HELPER: builds the reference image for the bounding box drawing tests.
    # Inputs: image is a three-dimensional numeric array (height, width, depth); color is a
    # one-dimensional array whose length must equal the depth. Paints the outermost pixel ring
    # (first and last row and column) with the colour, modifying the given image array in place,
    # and returns that same array. Raises a value error when the depths differ.
    """Fill the border of the image.

    Args:
      image: Numpy array of shape [height, width, depth].
      color: Numpy color of shape [depth] and either contents RGB/RGBA.

    Returns:
      image of original shape with border filled with "color".

    Raises:
      ValueError: Depths of image and color don"t match.
    """
    height, width, depth = image.shape
    if depth != color.shape[0]:
      raise ValueError("Image (%d) and color (%d) depths must match." %
                       (depth, color.shape[0]))
    image[0:height, 0, 0:depth] = color
    image[0:height, width - 1, 0:depth] = color
    image[0, 0:width, 0:depth] = color
    image[height - 1, 0:width, 0:depth] = color
    return image

  def _testDrawBoundingBoxColorCycling(self,
                                       img,
                                       dtype=dtypes.float32,
                                       colors=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the colour-cycling tests below.
    # KNOWN ISSUE: for single-channel images the red column of the colour table is overwritten
    # with one in place, which also silently modifies a caller-supplied colour table array.
    # Inputs: img is a three-dimensional array (height, width, depth of 1, 3 or 4) used as the
    # background; dtype is the image element type for the op input (default 32-bit float);
    # colors is an optional table of RGBA rows, with the default being a hard-coded copy of the
    # op's built-in ten-colour table that must stay in sync with the C++ kernel.
    # For every box count from one to one more than the number of colours it draws that many
    # identical full-image boxes (all covering the whole image so they overlap), computes the
    # expected image by filling the border with the colour of the last box (index wraps around the
    # table), runs the op in a CPU-only session and asserts the two images are equal. Returns
    # nothing; asserts the image rank is three and the depth is supported.
    """Tests if cycling works appropriately.

    Args:
      img: 3-D numpy image on which to draw.
      dtype: image dtype (float, half).
      colors: color table.
    """
    color_table = colors
    if colors is None:
      # THIS TABLE MUST MATCH draw_bounding_box_op.cc
      color_table = np.asarray([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1],
                                [0, 1, 0, 1], [0.5, 0, 0.5,
                                               1], [0.5, 0.5, 0, 1],
                                [0.5, 0, 0, 1], [0, 0, 0.5, 1], [0, 1, 1, 1],
                                [1, 0, 1, 1]])
    assert len(img.shape) == 3
    depth = img.shape[2]
    assert depth <= color_table.shape[1]
    assert depth == 1 or depth == 3 or depth == 4
    ## Set red channel to 1 if image is GRY.
    if depth == 1:
      color_table[:, 0] = 1
    num_colors = color_table.shape[0]
    for num_boxes in range(1, num_colors + 2):
      # Generate draw_bounding_box_op drawn image
      image = np.copy(img)
      color = color_table[(num_boxes - 1) % num_colors, 0:depth]
      test_drawn_image = self._fillBorder(image, color)
      bboxes = np.asarray([0, 0, 1, 1])
      bboxes = np.vstack([bboxes for _ in range(num_boxes)])
      bboxes = math_ops.cast(bboxes, dtypes.float32)
      bboxes = array_ops.expand_dims(bboxes, 0)
      image = ops.convert_to_tensor(image)
      image = image_ops_impl.convert_image_dtype(image, dtype)
      image = array_ops.expand_dims(image, 0)
      image = image_ops.draw_bounding_boxes(image, bboxes, colors=colors)
      with self.cached_session(use_gpu=False) as sess:
        op_drawn_image = np.squeeze(sess.run(image), 0)
        self.assertAllEqual(test_drawn_image, op_drawn_image)

  def testDrawBoundingBoxRGBColorCycling(self):
    # [FUNCTION DOC]
    # Test: ten by ten three-channel image with the default colour table; runs the shared
    # colour-cycling check.
    """Test if RGB color cycling works correctly."""
    image = np.zeros([10, 10, 3], "float32")
    self._testDrawBoundingBoxColorCycling(image)

  def testDrawBoundingBoxRGBAColorCycling(self):
    # [FUNCTION DOC]
    # Test: ten by ten four-channel image with the default colour table; runs the shared
    # colour-cycling check.
    """Test if RGBA color cycling works correctly."""
    image = np.zeros([10, 10, 4], "float32")
    self._testDrawBoundingBoxColorCycling(image)

  def testDrawBoundingBoxGRY(self):
    # [FUNCTION DOC]
    # Test: four by four single-channel image with the default colour table (whose red column is
    # forced to one by the helper); runs the shared colour-cycling check.
    """Test if drawing bounding box on a GRY image works."""
    image = np.zeros([4, 4, 1], "float32")
    self._testDrawBoundingBoxColorCycling(image)

  def testDrawBoundingBoxRGBColorCyclingWithColors(self):
    # [FUNCTION DOC]
    # Test: ten by ten three-channel image with a user-supplied six-colour table; runs the shared
    # colour-cycling check so that cycling wraps at the custom table length.
    """Test if RGB color cycling works correctly with provided colors."""
    image = np.zeros([10, 10, 3], "float32")
    colors = np.asarray([[1, 1, 0, 1], [0, 0, 1, 1], [0.5, 0, 0.5, 1],
                         [0.5, 0.5, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]])
    self._testDrawBoundingBoxColorCycling(image, colors=colors)

  def testDrawBoundingBoxRGBAColorCyclingWithColors(self):
    # [FUNCTION DOC]
    # Test: ten by ten four-channel image with a user-supplied four-colour table; runs the shared
    # colour-cycling check so that cycling wraps at the custom table length.
    """Test if RGBA color cycling works correctly with provided colors."""
    image = np.zeros([10, 10, 4], "float32")
    colors = np.asarray([[0.5, 0, 0.5, 1], [0.5, 0.5, 0, 1], [0.5, 0, 0, 1],
                         [0, 0, 0.5, 1]])
    self._testDrawBoundingBoxColorCycling(image, colors=colors)

  def testDrawBoundingBoxHalf(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copied from the RGBA colour test and does not mention that
    # the point of this test is the half-precision image dtype.
    # Test: ten by ten four-channel image converted to half precision, with a user-supplied
    # four-colour table; runs the shared colour-cycling check with the half dtype.
    """Test if RGBA color cycling works correctly with provided colors."""
    image = np.zeros([10, 10, 4], "float32")
    colors = np.asarray([[0.5, 0, 0.5, 1], [0.5, 0.5, 0, 1], [0.5, 0, 0, 1],
                         [0, 0, 0.5, 1]])
    self._testDrawBoundingBoxColorCycling(
        image, dtype=dtypes.half, colors=colors)

  # generate_bound_box_proposals is only available on GPU.
  @test_util.run_gpu_only
  def testGenerateBoundingBoxProposals(self):
    # [FUNCTION DOC]
    # Test (runs only when a GPU is available): calls the proposal generation op with a score
    # tensor of the wrong rank and empty lists for the other inputs, and expects a value error or
    # invalid-argument error whose message says the input must be rank four.
    # Op only exists on GPU.
    with self.cached_session(use_gpu=True):
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "must be rank 4"):
        scores = constant_op.constant(
            value=[[[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]])
        self.evaluate(
            image_ops.generate_bounding_box_proposals(
                scores=scores,
                bbox_deltas=[],
                image_info=[],
                anchors=[],
                pre_nms_topn=1))

if __name__ == "__main__":
  test.main()
