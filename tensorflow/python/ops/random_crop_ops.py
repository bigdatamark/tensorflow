# [MODULE DOC]
# Public ops for randomly cropping a tensor to a given size: random_crop (uses the global/op
# seed) and stateless_random_crop (deterministic given an explicit two-element seed). Both check
# that the input is at least as large as the crop size in every dimension, draw a uniform random
# offset per dimension and slice the input. Exported under the image namespace (random_crop also
# under legacy v1 names). No import-time side effects beyond export registration.
#
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
"""Operations for random tensor cropping."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export("image.random_crop", v1=["image.random_crop", "random_crop"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_crop")
def random_crop(value, size, seed=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the offset is drawn as a uniform integer over the whole range of the size type
  # and then reduced by the remainder operation, so offsets are very slightly biased toward small
  # values (modulo bias) rather than being exactly uniform as the docstring says.
  # Randomly crops a tensor to the requested size. Inputs: value (tensor to crop), size (1-D
  # integer tensor with one entry per dimension of value, converted to 32-bit integers), seed
  # (optional integer used with the global seed) and name (optional).
  # Adds a runtime assertion that every dimension of value is at least the crop size (failing with
  # an invalid-argument error that prints both shapes), then picks a random offset in each
  # dimension (zero up to dimension minus size) and slices. Returns the cropped tensor with the
  # same rank as value and shape equal to size. Consumes random state; different calls give
  # different results.
  """Randomly crops a tensor to a given size.

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Example usage:

  >>> image = [[1, 2, 3], [4, 5, 6]]
  >>> result = tf.image.random_crop(value=image, size=(1, 3))
  >>> result.shape.as_list()
  [1, 3]

  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_crop`. Unlike using the `seed` param with
  `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same
  results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: Python integer. Used to create a random seed. See
      `tf.random.set_seed`
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  with ops.name_scope(name, "random_crop", [value, size]) as name:
    value = ops.convert_to_tensor(value, name="value")
    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
    shape = array_ops.shape(value)
    check = control_flow_assert.Assert(
        math_ops.reduce_all(shape >= size),
        ["Need value.shape >= size, got ", shape, size],
        summarize=1000)
    shape = control_flow_ops.with_dependencies([check], shape)
    limit = shape - size + 1
    offset = random_ops.random_uniform(
        array_ops.shape(shape),
        dtype=size.dtype,
        maxval=size.dtype.max,
        seed=seed) % limit
    return array_ops.slice(value, offset, size, name=name)


@tf_export("image.stateless_random_crop", v1=[])
@dispatch.add_dispatch_support
def stateless_random_crop(value, size, seed, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: same modulo bias as the stateful variant: the offset is a uniform draw over the
  # whole integer range reduced by the remainder operation, so it is not exactly uniform.
  # Deterministic variant of random crop. Inputs: value (tensor to crop), size (1-D integer
  # tensor with one entry per dimension, converted to 32-bit integers), seed (required, a
  # two-element tensor of 32 or 64-bit integers; only 32-bit under XLA) and name (optional).
  # Asserts at run time that every dimension of value is at least the crop size, draws offsets
  # with the stateless uniform generator, and slices. Returns the cropped tensor. The same seed
  # always gives the same crop, independent of global seed or call count; no random state is used.
  """Randomly crops a tensor to a given size in a deterministic manner.

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).

  Usage Example:

  >>> image = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_crop(value=image, size=(1, 2, 3), seed=seed)
  <tf.Tensor: shape=(1, 2, 3), dtype=int32, numpy=
  array([[[1, 2, 3],
          [4, 5, 6]]], dtype=int32)>

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  with ops.name_scope(name, "random_crop", [value, size]) as name:
    value = ops.convert_to_tensor(value, name="value")
    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
    shape = array_ops.shape(value)
    check = control_flow_assert.Assert(
        math_ops.reduce_all(shape >= size),
        ["Need value.shape >= size, got ", shape, size],
        summarize=1000)
    shape = control_flow_ops.with_dependencies([check], shape)
    limit = shape - size + 1
    offset = stateless_random_ops.stateless_random_uniform(
        array_ops.shape(shape),
        dtype=size.dtype,
        maxval=size.dtype.max,
        seed=seed) % limit
    return array_ops.slice(value, offset, size, name=name)
  