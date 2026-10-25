# [MODULE DOC]
# Registers ragged-tensor dispatchers for the image resize APIs (the version-2 and version-1
# resize functions). When the images argument is a ragged tensor, each image is converted to a
# dense tensor and resized individually through a map over the batch. Import-time side effect:
# registration of the two dispatch overrides. Not a public API module itself.
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
"""Image operations for RaggedTensors."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_api(image_ops.resize_images_v2)
def resize_images_v2(images: ragged_tensor.RaggedTensor,
                     size,
                     method=image_ops.ResizeMethod.BILINEAR,
                     preserve_aspect_ratio=False,
                     antialias=False,
                     name=None):
  # [FUNCTION DOC]
  # Dispatch override of the version-2 image resize for ragged inputs.
  # Inputs: images, a rank-4 ragged tensor (batch, height, width, channels); size, two integers
  # or a tensor giving new height and width; method, resize method (default bilinear);
  # preserve_aspect_ratio, bool; antialias, bool; name, optional scope name.
  # Output: a dense float tensor of resized images. Raises a value error (from the helper) if
  # the rank is not four. Opens a name scope named for ragged image resizing.
  """RaggedTensor dispatcher for tf.image.resize (tf-v2)."""
  with ops.name_scope(name, "RaggedResizeImages", [images, size]):
    return _resize_images(
        image_ops.resize_images_v2,
        images,
        size,
        method=method,
        preserve_aspect_ratio=preserve_aspect_ratio,
        antialias=antialias)


@dispatch.dispatch_for_api(image_ops.resize_images)
def resize_images_v1(images: ragged_tensor.RaggedTensor,
                     size,
                     method=image_ops.ResizeMethodV1.BILINEAR,
                     align_corners=False,
                     preserve_aspect_ratio=False,
                     name=None):
  # [FUNCTION DOC]
  # Dispatch override of the version-1 image resize for ragged inputs.
  # Inputs: images, a rank-4 ragged tensor; size, new height and width; method, version-1
  # resize method (default bilinear); align_corners, bool; preserve_aspect_ratio, bool;
  # name, optional scope name.
  # Output: a dense float tensor of resized images. Raises a value error (from the helper) if
  # the rank is not four. Opens a name scope named for ragged image resizing.
  """RaggedTensor dispatcher for tf.image.resize (tf-v1)."""
  with ops.name_scope(name, "RaggedResizeImages", [images, size]):
    return _resize_images(
        image_ops.resize_images,
        images,
        size,
        method=method,
        preserve_aspect_ratio=preserve_aspect_ratio,
        align_corners=align_corners)


def _resize_images(resize_op, images, size, **kwargs):
  # [FUNCTION DOC]
  # HELPER: shared implementation for the version-1 and version-2 ragged resize dispatchers.
  # Inputs: resize_op, the dense resize function to apply per image; images, rank-4 ragged
  # tensor; size, target height and width; extra keyword arguments passed to resize_op.
  # Output: dense tensor of resized images. Raises a value error if images is not rank four.
  # If the static batch size is zero, returns an empty float tensor of the right size and
  # channel count; if known and nonzero, maps over the batch; if unknown, uses a graph
  # conditional that returns the empty result when the batch has no rows.
  """RaggedTensor dispatcher for tf.image.resize."""
  if images.shape.rank != 4:
    raise ValueError(
        "tf.image.resize: images.shape.rank must be 4 if images is ragged.")

  # Determine the output shape (excluding the batch dimension).
  static_batch_size = tensor_shape.dimension_value(images.shape[0])
  size = ops.convert_to_tensor(size, dtypes.int32, "size")
  size_as_shape = tensor_util.constant_value_as_shape(size).with_rank(2)
  out_shape = size_as_shape + images.shape[-1:]
  out_spec = tensor_spec.TensorSpec(out_shape, dtypes.float32)

  def resize_one(image):
    # [FUNCTION DOC]
    # HELPER: resizes one image of the batch. Converts a ragged image to a dense tensor
    # (padding with default zeros), then applies the enclosing resize function with the target
    # size and extra keyword arguments. Returns the resized dense image.
    if isinstance(image, ragged_tensor.RaggedTensor):
      image = image.to_tensor()
    return resize_op(image, size, **kwargs)

  def resize_with_map():
    # [FUNCTION DOC]
    # HELPER: maps the per-image resize over the batch, with a float output signature of the
    # computed output shape. Returns the stacked dense result.
    return map_fn.map_fn_v2(resize_one, images, fn_output_signature=out_spec)

  def empty_result():
    # [FUNCTION DOC]
    # HELPER: builds the result for an empty batch: a float zeros tensor with zero images, the
    # target height and width, and the channel count taken from the flat values at run time.
    channels = array_ops.shape(images.flat_values)[-1:]
    return array_ops.zeros(array_ops.concat([[0], size, channels], axis=0))

  if static_batch_size == 0:
    return empty_result()
  elif static_batch_size is not None:
    return resize_with_map()
  else:
    empty_batch = math_ops.equal(images.nrows(), 0)
    return cond.cond(empty_batch, empty_result, resize_with_map)
