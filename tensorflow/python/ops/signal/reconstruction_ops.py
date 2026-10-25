# [MODULE DOC]
# Signal reconstruction by overlapped addition of frames.
# Exposes the public op signal.overlap_and_add, the inverse of framing: it sums frames that are
# offset by a hop size into one longer signal. Import-time side effect: registers the op with the
# public API exporter and dispatch support. Used by the inverse STFT in spectral_ops.
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
"""Signal reconstruction via overlapped addition of frames."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export("signal.overlap_and_add")
@dispatch.add_dispatch_support
def overlap_and_add(signal, frame_step, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says frame_step must be at most the frame length, but no such check
  # is made, so larger values are not rejected by this function.
  # Public op signal.overlap_and_add. Inputs: signal, a tensor of rank at least two shaped
  # outer dimensions, frames, frame_length; frame_step, an integer scalar (python or tensor); and
  # an optional name. Output: tensor of shape outer dimensions plus one output dimension of length
  # (frames - 1) * frame_step + frame_length, where overlapping frames are summed.
  # Raises a value error if the rank is below two, frame_step is not a scalar (shape error) or not
  # an integer type. When frame_step equals the frame length it just reshapes. Otherwise it pads,
  # reshapes and transposes the frames into segments, reduces by summation, flattens and truncates.
  # Uses static values where possible. Adds only graph ops.
  """Reconstructs a signal from a framed representation.

  Adds potentially overlapping frames of a signal with shape
  `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
  The resulting tensor has shape `[..., output_size]` where

      output_size = (frames - 1) * frame_step + frame_length

  Args:
    signal: A [..., frames, frame_length] `Tensor`. All dimensions may be
      unknown, and rank must be at least 2.
    frame_step: An integer or scalar `Tensor` denoting overlap offsets. Must be
      less than or equal to `frame_length`.
    name: An optional name for the operation.

  Returns:
    A `Tensor` with shape `[..., output_size]` containing the overlap-added
    frames of `signal`'s inner-most two dimensions.

  Raises:
    ValueError: If `signal`'s rank is less than 2, or `frame_step` is not a
      scalar integer.
  """
  with ops.name_scope(name, "overlap_and_add", [signal, frame_step]):
    signal = ops.convert_to_tensor(signal, name="signal")
    signal.shape.with_rank_at_least(2)
    frame_step = ops.convert_to_tensor(frame_step, name="frame_step")
    frame_step.shape.assert_has_rank(0)
    if not frame_step.dtype.is_integer:
      raise ValueError("frame_step must be an integer. Got %s" %
                       frame_step.dtype)
    frame_step_static = tensor_util.constant_value(frame_step)
    frame_step_is_static = frame_step_static is not None
    frame_step = frame_step_static if frame_step_is_static else frame_step

    signal_shape = array_ops.shape(signal)
    signal_shape_static = tensor_util.constant_value(signal_shape)
    if signal_shape_static is not None:
      signal_shape = signal_shape_static

    # All dimensions that are not part of the overlap-and-add. Can be empty for
    # rank 2 inputs.
    outer_dimensions = signal_shape[:-2]
    outer_rank = array_ops.size(outer_dimensions)
    outer_rank_static = tensor_util.constant_value(outer_rank)
    if outer_rank_static is not None:
      outer_rank = outer_rank_static

    def full_shape(inner_shape):
      # [FUNCTION DOC]
      # HELPER: shape builder for overlap_and_add. Concatenates the captured outer (batch)
      # dimensions with the given inner shape list and returns the full shape tensor.
      return array_ops.concat([outer_dimensions, inner_shape], 0)

    frame_length = signal_shape[-1]
    frames = signal_shape[-2]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # If frame_length is equal to frame_step, there's no overlap so just
    # reshape the tensor.
    if (frame_step_is_static and signal.shape.dims is not None and
        frame_step == signal.shape.dims[-1].value):
      output_shape = full_shape([output_length])
      return array_ops.reshape(signal, output_shape, name="fast_path")

    # The following code is documented using this example:
    #
    # frame_step = 2
    # signal.shape = (3, 5)
    # a b c d e
    # f g h i j
    # k l m n o

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)  # Divide and round up.

    # Pad the frame_length dimension to a multiple of the frame step.
    # Pad the frames dimension by `segments` so that signal.shape = (6, 6)
    # a b c d e 0
    # f g h i j 0
    # k l m n o 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    paddings = [[0, segments], [0, segments * frame_step - frame_length]]
    outer_paddings = array_ops.zeros([outer_rank, 2], dtypes.int32)
    paddings = array_ops.concat([outer_paddings, paddings], 0)
    signal = array_ops.pad(signal, paddings)

    # Reshape so that signal.shape = (3, 6, 2)
    # ab cd e0
    # fg hi j0
    # kl mn o0
    # 00 00 00
    # 00 00 00
    # 00 00 00
    shape = full_shape([frames + segments, segments, frame_step])
    signal = array_ops.reshape(signal, shape)

    # Transpose dimensions so that signal.shape = (3, 6, 2)
    # ab fg kl 00 00 00
    # cd hi mn 00 00 00
    # e0 j0 o0 00 00 00
    perm = array_ops.concat(
        [math_ops.range(outer_rank), outer_rank + [1, 0, 2]], 0)
    perm_static = tensor_util.constant_value(perm)
    perm = perm_static if perm_static is not None else perm
    signal = array_ops.transpose(signal, perm)

    # Reshape so that signal.shape = (18, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0 00 00 00
    shape = full_shape([(frames + segments) * segments, frame_step])
    signal = array_ops.reshape(signal, shape)

    # Truncate so that signal.shape = (15, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0
    signal = signal[..., :(frames + segments - 1) * segments, :]

    # Reshape so that signal.shape = (3, 5, 2)
    # ab fg kl 00 00
    # 00 cd hi mn 00
    # 00 00 e0 j0 o0
    shape = full_shape([segments, (frames + segments - 1), frame_step])
    signal = array_ops.reshape(signal, shape)

    # Now, reduce over the columns, to achieve the desired sum.
    signal = math_ops.reduce_sum(signal, -3)

    # Flatten the array.
    shape = full_shape([(frames + segments - 1) * frame_step])
    signal = array_ops.reshape(signal, shape)

    # Truncate to final length.
    signal = signal[..., :output_length]

    return signal
