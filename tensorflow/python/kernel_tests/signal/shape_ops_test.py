# [MODULE DOC]
# Unit tests for the signal framing op (shape_ops.frame). Covers index mapping with and without
# end padding, argument validation, zero-length signals, static shape inference, mono/stereo/
# multi-dimensional inputs, the axis argument, windows larger than the signal, dtype
# preservation, dynamic shapes, numerical gradients and grappler constant folding.
# Uses the shared signal test_util helper for graph optimization. No import-time side effects
# beyond the test runner entry point at the bottom.
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
"""Tests for shape_ops."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.kernel_tests.signal import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.platform import test


@tf_test_util.run_all_in_graph_and_eager_modes
class FrameTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering shape_ops.frame, which slices a signal into overlapping frames along an
  # axis. Decorated so every test runs in both graph and eager modes. Holds no shared fixtures.

  def test_mapping_of_indices_without_padding(self):
    # [FUNCTION DOC]
    # Frames a 1x9152 integer ramp with frame length 512, step 180 and no end padding, and
    # asserts the output equals the expected index matrix of 49 frames (each frame offset by
    # a multiple of the step). Returns nothing.
    tensor = constant_op.constant(np.arange(9152), dtypes.int32)
    tensor = array_ops.expand_dims(tensor, 0)

    result = shape_ops.frame(tensor, 512, 180, pad_end=False)

    expected = np.tile(np.arange(512), (49, 1))
    expected += np.tile(np.arange(49) * 180, (512, 1)).T

    expected = np.expand_dims(expected, axis=0)
    expected = np.array(expected, dtype=np.int32)
    self.assertAllEqual(expected, result)

  def test_mapping_of_indices_with_padding(self):
    # [FUNCTION DOC]
    # Frames a 1x10000 integer ramp with frame length 512, step 192 and end padding, and asserts
    # the result has 53 frames where indices at or beyond the signal length are replaced by the
    # default pad value zero. Returns nothing.
    tensor = constant_op.constant(np.arange(10000), dtypes.int32)
    tensor = array_ops.expand_dims(tensor, 0)

    result = shape_ops.frame(tensor, 512, 192, pad_end=True)

    expected = np.tile(np.arange(512), (53, 1))
    expected += np.tile(np.arange(53) * 192, (512, 1)).T

    expected[expected >= 10000] = 0

    expected = np.expand_dims(expected, axis=0)
    expected = np.array(expected, dtype=np.int32)

    self.assertAllEqual(expected, result)

  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Asserts that frame raises a value error for a rank-0 signal, a non-scalar frame length,
    # a non-scalar frame step, and a non-scalar pad value. In graph mode additionally checks
    # that a signal of unknown rank does not raise. Returns nothing.
    # Rank 0 input signal.
    with self.assertRaises(ValueError):
      shape_ops.frame(1, 1, 1)

    if not context.executing_eagerly():
      # If the rank is unknown, do not raise an exception.
      shape_ops.frame(array_ops.placeholder_with_default(
          1, shape=tensor_shape.TensorShape(None)), 1, 1)

    # Non-scalar frame_length.
    with self.assertRaises(ValueError):
      shape_ops.frame([1], [1], 1)

    # Non-scalar frame_step.
    with self.assertRaises(ValueError):
      shape_ops.frame([1], 1, [1])

    # Non-scalar pad_value.
    with self.assertRaises(ValueError):
      shape_ops.frame([1], 1, 1, pad_end=True, pad_value=[1])

  def test_length_zero(self):
    # [FUNCTION DOC]
    # Frames an empty float signal with frame length 2 and step 1, both with and without end
    # padding, and asserts the evaluated result has shape zero frames by two samples.
    # Returns nothing.
    signal = constant_op.constant([], dtype=dtypes.float32)
    frame_length = 2
    frame_step = 1

    result = self.evaluate(shape_ops.frame(
        signal, frame_length, frame_step, pad_end=True, pad_value=99))
    self.assertEqual((0, 2), result.shape)

    result = self.evaluate(
        shape_ops.frame(signal, frame_length, frame_step, pad_end=False))
    self.assertEqual((0, 2), result.shape)

  def test_shape_inference(self):
    # [FUNCTION DOC]
    # Graph-mode only (returns immediately when eager). Checks static shape inference of frame
    # for known and unknown frame length, partially unknown input shapes, different axes and
    # padding modes, and finally that the inferred static shape matches the evaluated shape
    # for small signal lengths, frame lengths and both padding settings. Returns nothing.
    if context.executing_eagerly():
      return
    signal = array_ops.zeros((1, 1), dtype=dtypes.int32)
    frame_length = 2
    frame_step = 1
    # Shape inference is able to detect the rank and inner-most dimension
    # if frame_length is known at graph definition time.
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99)
    self.assertEqual([1, 1, 2], result.shape.as_list())

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False)
    self.assertEqual([1, 0, 2], result.shape.as_list())

    # If frame_length is not known, rank and (known) outer and inner dimensions
    # are inferred.
    signal = array_ops.zeros([1, 2, 3, 4], dtype=dtypes.int32)
    frame_length = array_ops.placeholder_with_default(
        ops.convert_to_tensor(0, dtypes.int32), shape=[])
    frame_step = 1
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99, axis=1)
    self.assertEqual([1, 2, None, 3, 4], result.shape.as_list())

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False, axis=1)
    self.assertEqual([1, None, None, 3, 4], result.shape.as_list())

    # If frame_length and inner-most dimension is known, rank, inner dimensions,
    # and known outer dimensions are inferred.
    signal = array_ops.placeholder_with_default(
        array_ops.zeros((0, 5, 0, 20, 5, 3), dtype=dtypes.int32),
        shape=[None, 5, None, 20, 5, 3])
    frame_length = 4
    frame_step = 3
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99, axis=3)
    self.assertEqual([None, 5, None, 7, 4, 5, 3], result.shape.as_list())

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False, axis=3)
    self.assertEqual([None, 5, None, 6, 4, 5, 3], result.shape.as_list())

    # Test that shape inference is consistent with actual returned shapes for
    # small values of signal_length, frame_length, frame_step, and pad_end in
    # [True, False].
    frame_step = 1
    for signal_length in range(2):
      signal = [0] * signal_length
      for frame_length in range(2):
        for pad_end in [False, True]:
          op = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=pad_end, pad_value=99)
          result = self.evaluate(op)
          self.assertEqual(op.shape.as_list(), list(result.shape))

  def test_basic_mono(self):
    # [FUNCTION DOC]
    # Frames a six-sample signal (length 3, step 2) reshaped with 0 to 4 leading singleton
    # dimensions. With padding the last frame is filled with 99; without padding the partial
    # last frame is dropped. Asserts exact frame contents. Returns nothing.
    signal = np.arange(6)
    frame_length = 3
    frame_step = 2

    for rank in range(5):
      nd_signal = np.reshape(signal, (1,) * rank + signal.shape)

      # With padding, we pad the last frame with pad_value.
      result = shape_ops.frame(nd_signal, frame_length, frame_step,
                               pad_end=True, pad_value=99)
      expected_inner_frames = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 99]])
      expected = np.reshape(
          expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
      self.assertAllEqual(expected, result)

      # Without padding, we drop the last frame.
      expected_inner_frames = np.array([[0, 1, 2], [2, 3, 4]])
      expected = np.reshape(
          expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
      result = shape_ops.frame(nd_signal, frame_length, frame_step,
                               pad_end=False)
      self.assertAllEqual(expected, result)

  def test_basic_stereo(self):
    # [FUNCTION DOC]
    # Same as the mono test but with a two-channel signal (rows 0..5 and 10..15), checking that
    # framing is applied independently per channel with and without end padding (pad value 99),
    # across 0 to 4 leading singleton dimensions. Returns nothing.
    signal = np.vstack([np.arange(6),
                        np.arange(6) + 10])
    frame_length = 3
    frame_step = 2

    for rank in range(5):
      nd_signal = np.reshape(signal, (1,) * rank + signal.shape)

      # With padding, we pad the last frame with pad_value.
      result = shape_ops.frame(nd_signal, frame_length, frame_step,
                               pad_end=True, pad_value=99)
      expected_inner_frames = np.array([
          [[0, 1, 2], [2, 3, 4], [4, 5, 99]],
          [[10, 11, 12], [12, 13, 14], [14, 15, 99]]])
      expected = np.reshape(
          expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
      self.assertAllEqual(expected, result)

      # Without padding, we drop the last frame.
      expected_inner_frames = np.array([[[0, 1, 2], [2, 3, 4]],
                                        [[10, 11, 12], [12, 13, 14]]])
      expected = np.reshape(
          expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
      result = shape_ops.frame(nd_signal, frame_length, frame_step,
                               pad_end=False)
      self.assertAllEqual(expected, result)

  def test_complex_shape(self):
    # [FUNCTION DOC]
    # Frames a signal of shape 2x1x3x1x6 (length 3, step 2) with and without padding and
    # asserts the nested expected result, confirming leading dimensions are preserved and the
    # last axis is expanded into frames. Returns nothing.
    signal = np.vstack([np.arange(6),
                        np.arange(6) + 10,
                        np.arange(6) + 20,
                        np.arange(6) + 30,
                        np.arange(6) + 40,
                        np.arange(6) + 50])
    signal = np.reshape(signal, (2, 1, 3, 1, 6))
    frame_length = 3
    frame_step = 2

    # With padding, we pad the last frame with pad_value.
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99)
    # Resulting shape is (2, 1, 3, 1, 3, 3).
    expected = [[[[[[0, 1, 2], [2, 3, 4], [4, 5, 99]]],
                  [[[10, 11, 12], [12, 13, 14], [14, 15, 99]]],
                  [[[20, 21, 22], [22, 23, 24], [24, 25, 99]]]]],
                [[[[[30, 31, 32], [32, 33, 34], [34, 35, 99]]],
                  [[[40, 41, 42], [42, 43, 44], [44, 45, 99]]],
                  [[[50, 51, 52], [52, 53, 54], [54, 55, 99]]]]]]
    self.assertAllEqual(expected, result)

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False)
    # Resulting shape is (2, 1, 3, 1, 3, 2).
    expected = [[[[[[0, 1, 2], [2, 3, 4]]],
                  [[[10, 11, 12], [12, 13, 14]]],
                  [[[20, 21, 22], [22, 23, 24]]]]],
                [[[[[30, 31, 32], [32, 33, 34]]],
                  [[[40, 41, 42], [42, 43, 44]]],
                  [[[50, 51, 52], [52, 53, 54]]]]]]
    self.assertAllEqual(expected, result)

  def test_axis(self):
    # [FUNCTION DOC]
    # Frames a 2x4x2 signal along axis 1 with several frame lengths and steps (padding
    # enabled) and asserts the evaluated results, including zero padding of trailing frames.
    # Returns nothing.
    signal = np.reshape(np.arange(16), (2, 4, 2))
    result = shape_ops.frame(signal, frame_length=2, frame_step=2,
                             pad_end=True, axis=1)
    expected = np.reshape(np.arange(16), (2, 2, 2, 2))
    self.assertAllEqual(expected, self.evaluate(result))

    result = shape_ops.frame(signal, frame_length=2, frame_step=1,
                             pad_end=True, axis=1)
    expected = [[[[0, 1], [2, 3]],
                 [[2, 3], [4, 5]],
                 [[4, 5], [6, 7]],
                 [[6, 7], [0, 0]]],
                [[[8, 9], [10, 11]],
                 [[10, 11], [12, 13]],
                 [[12, 13], [14, 15]],
                 [[14, 15], [0, 0]]]]
    self.assertAllEqual(expected, self.evaluate(result))

    result = shape_ops.frame(signal, frame_length=3, frame_step=1,
                             pad_end=True, axis=1)
    expected = [[[[0, 1], [2, 3], [4, 5]],
                 [[2, 3], [4, 5], [6, 7]],
                 [[4, 5], [6, 7], [0, 0]],
                 [[6, 7], [0, 0], [0, 0]]],
                [[[8, 9], [10, 11], [12, 13]],
                 [[10, 11], [12, 13], [14, 15]],
                 [[12, 13], [14, 15], [0, 0]],
                 [[14, 15], [0, 0], [0, 0]]]]
    self.assertAllEqual(expected, self.evaluate(result))

  def test_window_larger_than_signal(self):
    # [FUNCTION DOC]
    # Frames a 2x2 float signal with frame length 4 (larger than the signal), steps 1 and 2.
    # With padding, asserts padded frames filled with 99; without padding, asserts an empty
    # frame dimension (shape 2x0x4). Returns nothing.
    signal = constant_op.constant([[1, 2], [11, 12]], dtype=dtypes.float32)
    frame_length = 4
    frame_step = 1

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99)
    self.assertAllClose([[[1, 2, 99, 99], [2, 99, 99, 99]],
                         [[11, 12, 99, 99], [12, 99, 99, 99]]], result)

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False)
    self.assertEqual((2, 0, 4), result.shape)

    frame_step = 2
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99)
    self.assertAllClose([[[1, 2, 99, 99]], [[11, 12, 99, 99]]], result)

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False)
    self.assertEqual((2, 0, 4), result.shape)

  def test_preserves_type(self):
    # [FUNCTION DOC]
    # Frames a float64 range tensor and asserts the output dtype equals the input dtype.
    # Returns nothing.
    signal = math_ops.range(10, dtype=dtypes.float64)
    frame_length = 2
    frame_step = 3

    result = shape_ops.frame(signal, frame_length, frame_step)
    self.assertEqual(result.dtype, signal.dtype)

  def test_dynamic_tensor(self):
    # [FUNCTION DOC]
    # Graph-mode only. Frames a placeholder whose dimensions are unknown at graph construction
    # time (fed a 3x4 array, length 2, step 2) and asserts the evaluated frames. Returns nothing.
    if context.executing_eagerly():
      return
    # Show that frame works even when the dimensions of its input are
    # not known at graph creation time.
    input_signal = np.vstack([np.arange(4), np.arange(4) + 10,
                              np.arange(4) + 20])
    frame_length = 2
    frame_step = 2

    signal_placeholder = array_ops.placeholder_with_default(
        input_signal, shape=(None, None))
    result = self.evaluate(
        shape_ops.frame(signal_placeholder, frame_length, frame_step))
    self.assertAllEqual([[[0, 1], [2, 3]],
                         [[10, 11], [12, 13]],
                         [[20, 21], [22, 23]]], result)

  def test_gradient_numerical(self):
    # [FUNCTION DOC]
    # Graph-mode only. Within a session, compares the analytic and numerical gradient of frame
    # (signal of ones, shape 2x128, length 33, step 9) and asserts the error is below 2e-5.
    # Returns nothing.
    if context.executing_eagerly():
      return
    with self.session():
      signal_shape = (2, 128)
      signal = array_ops.ones(signal_shape)
      frame_length = 33
      frame_step = 9
      frames = shape_ops.frame(signal, frame_length, frame_step)
      error = test.compute_gradient_error(
          signal, signal_shape, frames, frames.shape.as_list())
      self.assertLess(error, 2e-5)

  def test_constant_folding(self):
    # [FUNCTION DOC]
    # Graph-mode only. For both padding settings, builds a graph framing a constant tensor of
    # ones, runs the grappler optimizer via the test_util helper and asserts the rewritten graph
    # collapses to a single node, i.e. frame is constant-foldable. Returns nothing.
    """frame should be constant foldable for constant inputs."""
    if context.executing_eagerly():
      return
    for pad_end in [True, False]:
      g = ops.Graph()
      with g.as_default():
        frame_length, frame_step = 32, 16
        signal_shape = (2, 128)
        signal = array_ops.ones(signal_shape)
        frames = shape_ops.frame(signal, frame_length, frame_step,
                                 pad_end=pad_end)
        rewritten_graph = test_util.grappler_optimize(g, [frames])
        self.assertEqual(1, len(rewritten_graph.node))


if __name__ == "__main__":
  test.main()
