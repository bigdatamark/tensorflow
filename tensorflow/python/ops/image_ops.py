# [MODULE DOC]
# Public image ops module (documented as the tf.image API). It mostly re-exports everything from
# the generated image op wrappers and from the image implementation module through wildcard
# imports, plus two legacy private helpers kept for internal users. It also defines helpers that
# convert between flat eight-number projective transforms and 3x3 matrices, and registers the
# gradient functions for the projective transform ops (versions 2 and 3). Import-time side effects:
# gradient registration. The module defines a private set of supported image data types used by
# those gradients.
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

"""Image ops.

The `tf.image` module contains various functions for image
processing and decoding-encoding Ops.

Many of the encoding/decoding functions are also available in the
core `tf.io` module.

## Image processing

### Resizing

The resizing Ops accept input images as tensors of several types. They always
output resized images as float32 tensors.

The convenience function `tf.image.resize` supports both 4-D
and 3-D tensors as input and output.  4-D tensors are for batches of images,
3-D tensors for individual images.

Resized images will be distorted if their original aspect ratio is not the
same as size. To avoid distortions see tf.image.resize_with_pad.

*   `tf.image.resize`
*   `tf.image.resize_with_pad`
*   `tf.image.resize_with_crop_or_pad`

The Class `tf.image.ResizeMethod` provides various resize methods like
`bilinear`, `nearest_neighbor`.

### Converting Between Colorspaces

Image ops work either on individual images or on batches of images, depending on
the shape of their input Tensor.

If 3-D, the shape is `[height, width, channels]`, and the Tensor represents one
image. If 4-D, the shape is `[batch_size, height, width, channels]`, and the
Tensor represents `batch_size` images.

Currently, `channels` can usefully be 1, 2, 3, or 4. Single-channel images are
grayscale, images with 3 channels are encoded as either RGB or HSV. Images
with 2 or 4 channels include an alpha channel, which has to be stripped from the
image before passing the image to most image processing functions (and can be
re-attached later).

Internally, images are either stored in as one `float32` per channel per pixel
(implicitly, values are assumed to lie in `[0,1)`) or one `uint8` per channel
per pixel (values are assumed to lie in `[0,255]`).

TensorFlow can convert between images in RGB or HSV or YIQ.

*   `tf.image.rgb_to_grayscale`, `tf.image.grayscale_to_rgb`
*   `tf.image.rgb_to_hsv`, `tf.image.hsv_to_rgb`
*   `tf.image.rgb_to_yiq`, `tf.image.yiq_to_rgb`
*   `tf.image.rgb_to_yuv`, `tf.image.yuv_to_rgb`
*   `tf.image.image_gradients`
*   `tf.image.convert_image_dtype`

### Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness,
contrast, hue, and saturation.  Each adjustment can be done with predefined
parameters or with random parameters picked from predefined intervals. Random
adjustments are often useful to expand a training set and reduce overfitting.

If several adjustments are chained it is advisable to minimize the number of
redundant conversions by first converting the images to the most natural data
type and representation.

*   `tf.image.adjust_brightness`
*   `tf.image.adjust_contrast`
*   `tf.image.adjust_gamma`
*   `tf.image.adjust_hue`
*   `tf.image.adjust_jpeg_quality`
*   `tf.image.adjust_saturation`
*   `tf.image.random_brightness`
*   `tf.image.random_contrast`
*   `tf.image.random_hue`
*   `tf.image.random_saturation`
*   `tf.image.per_image_standardization`

### Working with Bounding Boxes

*   `tf.image.draw_bounding_boxes`
*   `tf.image.combined_non_max_suppression`
*   `tf.image.generate_bounding_box_proposals`
*   `tf.image.non_max_suppression`
*   `tf.image.non_max_suppression_overlaps`
*   `tf.image.non_max_suppression_padded`
*   `tf.image.non_max_suppression_with_scores`
*   `tf.image.pad_to_bounding_box`
*   `tf.image.sample_distorted_bounding_box`

### Cropping

*   `tf.image.central_crop`
*   `tf.image.crop_and_resize`
*   `tf.image.crop_to_bounding_box`
*   `tf.io.decode_and_crop_jpeg`
*   `tf.image.extract_glimpse`
*   `tf.image.random_crop`
*   `tf.image.resize_with_crop_or_pad`

### Flipping, Rotating and Transposing

*   `tf.image.flip_left_right`
*   `tf.image.flip_up_down`
*   `tf.image.random_flip_left_right`
*   `tf.image.random_flip_up_down`
*   `tf.image.rot90`
*   `tf.image.transpose`

## Image decoding and encoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats.  Encoded
images are represented by scalar string Tensors, decoded images by 3-D uint8
tensors of shape `[height, width, channels]`. (PNG also supports uint16.)

Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`

The encode and decode Ops apply to one image at a time.  Their input and output
are all of variable size.  If you need fixed size images, pass the output of
the decode Ops to one of the cropping and resizing Ops.

*   `tf.io.decode_bmp`
*   `tf.io.decode_gif`
*   `tf.io.decode_image`
*   `tf.io.decode_jpeg`
*   `tf.io.decode_and_crop_jpeg`
*   `tf.io.decode_png`
*   `tf.io.encode_jpeg`
*   `tf.io.encode_png`

API docstring: tensorflow.image
"""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import linalg_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_image_ops import *
from tensorflow.python.ops.image_ops_impl import *
# pylint: enable=wildcard-import

# TODO(drpng): remove these once internal use has discontinued.
# pylint: disable=unused-import
from tensorflow.python.ops.image_ops_impl import _Check3DImage
from tensorflow.python.ops.image_ops_impl import _ImageDimensions
# pylint: enable=unused-import

_IMAGE_DTYPES = frozenset([
    dtypes.uint8, dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32,
    dtypes.float64
])


def flat_transforms_to_matrices(transforms):
  # [FUNCTION DOC]
  # Converts projective transforms given as flat vectors of eight numbers into 3x3 matrices.
  # Input transforms: a vector of length 8 or a batch of shape N by 8 (anything convertible to a
  # tensor). Raises a value error if its rank is not 1 or 2. Reshapes to N by 8, appends a column
  # of ones as the implicit ninth entry and reshapes to N by 3 by 3. Returns that tensor of
  # matrices (always batched, even for a single input). Matrices map output coordinates to input
  # coordinates. Creates a name scope; no other side effects.
  """Converts `tf.contrib.image` projective transforms to affine matrices.

  Note that the output matrices map output coordinates to input coordinates. For
  the forward transformation matrix, call `tf.linalg.inv` on the result.

  Args:
    transforms: Vector of length 8, or batches of transforms with shape `(N,
      8)`.

  Returns:
    3D tensor of matrices with shape `(N, 3, 3)`. The output matrices map the
      *output coordinates* (in homogeneous coordinates) of each transform to the
      corresponding *input coordinates*.

  Raises:
    ValueError: If `transforms` have an invalid shape.
  """
  with ops.name_scope("flat_transforms_to_matrices"):
    transforms = ops.convert_to_tensor(transforms, name="transforms")
    if transforms.shape.ndims not in (1, 2):
      raise ValueError("Transforms should be 1D or 2D, got: %s" % transforms)
    # Make the transform(s) 2D in case the input is a single transform.
    transforms = array_ops.reshape(transforms, constant_op.constant([-1, 8]))
    num_transforms = array_ops.shape(transforms)[0]
    # Add a column of ones for the implicit last entry in the matrix.
    return array_ops.reshape(
        array_ops.concat(
            [transforms, array_ops.ones([num_transforms, 1])], axis=1),
        constant_op.constant([-1, 3, 3]))


def matrices_to_flat_transforms(transform_matrices):
  # [FUNCTION DOC]
  # Inverse of the flat-to-matrix conversion: turns 3x3 affine/projective matrices into flat
  # vectors of eight numbers. Input transform_matrices: one matrix (3x3) or a batch (N by 3 by 3).
  # Raises a value error if its rank is not 2 or 3. Flattens each matrix to nine entries, divides
  # by the last entry (normalizing it to one; no check for zero, which would give non-finite
  # values) and drops that entry. Returns a tensor of shape N by 8 (a single matrix yields a
  # one-row result). Creates a name scope; no other side effects.
  """Converts affine matrices to `tf.contrib.image` projective transforms.

  Note that we expect matrices that map output coordinates to input coordinates.
  To convert forward transformation matrices, call `tf.linalg.inv` on the
  matrices and use the result here.

  Args:
    transform_matrices: One or more affine transformation matrices, for the
      reverse transformation in homogeneous coordinates. Shape `(3, 3)` or `(N,
      3, 3)`.

  Returns:
    2D tensor of flat transforms with shape `(N, 8)`, which may be passed into
      `tf.contrib.image.transform`.

  Raises:
    ValueError: If `transform_matrices` have an invalid shape.
  """
  with ops.name_scope("matrices_to_flat_transforms"):
    transform_matrices = ops.convert_to_tensor(
        transform_matrices, name="transform_matrices")
    if transform_matrices.shape.ndims not in (2, 3):
      raise ValueError("Matrices should be 2D or 3D, got: %s" %
                       transform_matrices)
    # Flatten each matrix.
    transforms = array_ops.reshape(transform_matrices,
                                   constant_op.constant([-1, 9]))
    # Divide each matrix by the last entry (normally 1).
    transforms /= transforms[:, 8:9]
    return transforms[:, :8]


@ops.RegisterGradient("ImageProjectiveTransformV2")
def _image_projective_transform_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the version 2 projective transform op.
  # KNOWN ISSUE: the docstring says it is the gradient for the plain projective transform op; it is
  # registered only for version 2 and is nearly a duplicate of the version 3 gradient function.
  # Inputs: op (the forward operation, read for its image and transform inputs and the
  # interpolation and fill mode attributes) and grad (upstream gradient). Validates the image
  # dtype is in the supported set (raises a type error otherwise) and that transforms have rank 1
  # or 2 (type error otherwise). Inverts the transforms (through matrices and the matrix inverse),
  # applies the version 2 projective transform to grad with output size equal to the original
  # image height and width, and returns a list of three: that gradient for the images and null
  # values for the transforms and output shape inputs.
  """Computes the gradient for ImageProjectiveTransform."""
  images = op.inputs[0]
  transforms = op.inputs[1]
  interpolation = op.get_attr("interpolation")
  fill_mode = op.get_attr("fill_mode")

  image_or_images = ops.convert_to_tensor(images, name="images")
  transform_or_transforms = ops.convert_to_tensor(
      transforms, name="transforms", dtype=dtypes.float32)

  if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
    raise TypeError("Invalid dtype %s." % image_or_images.dtype)
  if len(transform_or_transforms.get_shape()) == 1:
    transforms = transform_or_transforms[None]
  elif len(transform_or_transforms.get_shape()) == 2:
    transforms = transform_or_transforms
  else:
    raise TypeError("Transforms should have rank 1 or 2.")

  # Invert transformations
  transforms = flat_transforms_to_matrices(transforms=transforms)
  inverse = linalg_ops.matrix_inverse(transforms)
  transforms = matrices_to_flat_transforms(inverse)
  output = gen_image_ops.image_projective_transform_v2(
      images=grad,
      transforms=transforms,
      output_shape=array_ops.shape(image_or_images)[1:3],
      interpolation=interpolation,
      fill_mode=fill_mode)
  return [output, None, None]


@ops.RegisterGradient("ImageProjectiveTransformV3")
def _image_projective_transform_v3_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the version 3 projective transform op.
  # KNOWN ISSUE: the docstring does not mention version 3, and the fill value is hard-coded to zero
  # in the backward pass regardless of the forward op's fill value input; the gradient with respect
  # to the fill value is reported as null.
  # Inputs: op (forward operation: image, transform inputs, interpolation and fill mode attributes)
  # and grad (upstream gradient). Raises a type error for an unsupported image dtype or transforms
  # of rank other than 1 or 2. Inverts the transforms, applies the version 3 projective transform
  # to grad with the original image height and width as output size, and returns a list of four:
  # the image gradient and null values for transforms, output shape and fill value.
  """Computes the gradient for ImageProjectiveTransform."""
  images = op.inputs[0]
  transforms = op.inputs[1]
  interpolation = op.get_attr("interpolation")
  fill_mode = op.get_attr("fill_mode")

  image_or_images = ops.convert_to_tensor(images, name="images")
  transform_or_transforms = ops.convert_to_tensor(
      transforms, name="transforms", dtype=dtypes.float32)

  if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
    raise TypeError("Invalid dtype %s." % image_or_images.dtype)
  if len(transform_or_transforms.get_shape()) == 1:
    transforms = transform_or_transforms[None]
  elif len(transform_or_transforms.get_shape()) == 2:
    transforms = transform_or_transforms
  else:
    raise TypeError("Transforms should have rank 1 or 2.")

  # Invert transformations
  transforms = flat_transforms_to_matrices(transforms=transforms)
  inverse = linalg_ops.matrix_inverse(transforms)
  transforms = matrices_to_flat_transforms(inverse)
  output = gen_image_ops.image_projective_transform_v3(
      images=grad,
      transforms=transforms,
      output_shape=array_ops.shape(image_or_images)[1:3],
      interpolation=interpolation,
      fill_mode=fill_mode,
      fill_value=0.0)
  return [output, None, None, None]
