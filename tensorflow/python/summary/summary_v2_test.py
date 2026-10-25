# [MODULE DOC]
# KNOWN ISSUE: the availability flag is read from the tb_summary module with a safe lookup, but that
# module never defines it, so the flag is always false and every test in this module is skipped.
# Test module for the version 1 summary functions when TF2 behavior is enabled: in eager mode they
# forward to the version 2 TensorBoard summary functions (replaced by mocks here), emit a warning
# and skip when no default writer or global step exists, and return an empty string tensor. Also
# checks that the version 2 write functions reject non-scalar steps. Skipped entirely when the
# TensorBoard package is unavailable. Run as a test binary.
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
"""Tests for the API surface of the V1 tf.summary ops when TF2 is enabled.

V1 summary ops will invoke V2 TensorBoard summary ops in eager mode.
"""

import unittest

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.summary import tb_summary
from tensorflow.python.training import training_util

TENSORBOARD_AVAILABLE = (
    hasattr(tb_summary, 'TENSORBOARD_AVAILABLE')
    and tb_summary.TENSORBOARD_AVAILABLE
)


@unittest.skipIf(not TENSORBOARD_AVAILABLE, 'Tensorboard not installed.')
class SummaryV2Test(test.TestCase):
  # [CLASS DOC]
  # Test case (skipped if TensorBoard is not installed) for the bridging between version 1
  # summary functions and version 2 summary functions. Tests run only under TF2 behavior, replace
  # the TensorBoard summary function with an auto-specced mock, write with a summary file writer
  # in the temp directory, and assert on the call arguments and returned tensor.

  @test_util.run_v2_only
  def test_scalar_summary_v2__w_writer(self):
    # [FUNCTION DOC]
    # Test (TF2 only): with a default writer at step one, the version 1 scalar function returns an
    # empty string tensor and calls the mocked version 2 scalar exactly once with the given name,
    # data and the step.
    """Tests scalar v2 invocation with a v2 writer."""
    with test.mock.patch.object(
        tb_summary, 'scalar', autospec=True) as mock_scalar_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=1):
        i = constant_op.constant(2.5)
        tensor = summary_lib.scalar('float', i)
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_scalar_v2.assert_called_once_with(name='float', data=i, step=1)

  @test_util.run_v2_only
  def test_scalar_summary_v2__wo_writer(self):
    # [FUNCTION DOC]
    # Test (TF2 only): without any default writer the version 1 scalar function emits a user
    # warning that no default summary writer was found and does not call the version 2 scalar.
    """Tests scalar v2 invocation with no writer."""
    with self.assertWarnsRegex(
        UserWarning, 'default summary writer not found'):
      with test.mock.patch.object(
          tb_summary, 'scalar', autospec=True) as mock_scalar_v2:
        summary_lib.scalar('float', constant_op.constant(2.5))
    mock_scalar_v2.assert_not_called()

  @test_util.run_v2_only
  def test_scalar_summary_v2__global_step_not_set(self):
    # [FUNCTION DOC]
    # Test (TF2 only): with a default writer but no step set, the version 1 scalar function warns
    # that the global step is not set and does not call the version 2 scalar.
    """Tests scalar v2 invocation when global step is not set."""
    with self.assertWarnsRegex(UserWarning, 'global step not set'):
      with test.mock.patch.object(
          tb_summary, 'scalar', autospec=True) as mock_scalar_v2:
        with summary_ops_v2.create_summary_file_writer(
            self.get_temp_dir()).as_default():
          summary_lib.scalar('float', constant_op.constant(2.5))
    mock_scalar_v2.assert_not_called()

  @test_util.run_v2_only
  def test_tf_summary_scalar_invalid_step_shapes_raise(self):
    # [FUNCTION DOC]
    # Test (TF2 only): with a writer active, writing a scalar with each of several invalid steps
    # (empty or multi-element tuples and lists, rank one and rank two tensors) raises a value
    # error every time.
    writer = summary_ops_v2.create_summary_file_writer(self.get_temp_dir())

    invalid_steps = [
        (),  # empty tuple
        [],  # empty list
        [1, 2],  # non-scalar list
        (1, 2),  # non-scalar tuple
        array_ops.ones((2,)),  # rank-1 tensor
        array_ops.ones((1, 1)),  # rank-2 tensor
    ]

    with writer.as_default(step=1):
      for step in invalid_steps:
        with self.assertRaises(ValueError):
          summary_ops_v2.scalar('loss', 0.5, step=step)

  @test_util.run_v2_only
  def test_tf_summary_histogram_invalid_step_raises(self):
    # [FUNCTION DOC]
    # Test (TF2 only): writing a histogram with an empty tuple as step raises a value error.
    writer = summary_ops_v2.create_summary_file_writer(self.get_temp_dir())

    with writer.as_default(step=1):
      with self.assertRaises(ValueError):
        summary_ops_v2.histogram(
            'h',
            array_ops.ones((10,)),
            step=(),
        )

  @test_util.run_v2_only
  def test_tf_summary_image_invalid_step_raises(self):
    # [FUNCTION DOC]
    # Test (TF2 only): writing an image summary with an empty tuple as step raises a value error.
    writer = summary_ops_v2.create_summary_file_writer(self.get_temp_dir())

    image = array_ops.ones((1, 4, 4, 3))

    with writer.as_default(step=1):
      with self.assertRaises(ValueError):
        summary_ops_v2.image(
            'img',
            image,
            step=(),
        )

  @test_util.run_v2_only
  def test_tf_summary_audio_invalid_step_raises(self):
    # [FUNCTION DOC]
    # Test (TF2 only): writing an audio summary (sample rate sixteen thousand, one output) with an
    # empty tuple as step raises a value error.
    writer = summary_ops_v2.create_summary_file_writer(self.get_temp_dir())

    audio = array_ops.ones((1, 16000, 1))

    with writer.as_default(step=1):
      with self.assertRaises(ValueError):
        summary_ops_v2.audio(
            'audio',
            audio,
            sample_rate=16000,
            max_outputs=1,
            step=(),
        )

  @test_util.run_v2_only
  def test_tf_summary_write_invalid_step_raises(self):
    # [FUNCTION DOC]
    # Test (TF2 only): the generic write function with an empty tuple as step raises a value
    # error.
    writer = summary_ops_v2.create_summary_file_writer(self.get_temp_dir())

    with writer.as_default(step=1):
      with self.assertRaises(ValueError):
        summary_ops_v2.write(
            tag='tag',
            tensor=constant_op.constant(1.0),
            step=(),
        )

  @test_util.run_v2_only
  def test_tf_summary_trace_export_invalid_step_raises(self):
    # [FUNCTION DOC]
    # Test (TF2 only): turns graph tracing on under a default writer, then exporting the trace
    # with an empty tuple as step raises a value error. Side effect: tracing is left on.
    writer = summary_ops_v2.create_summary_file_writer(self.get_temp_dir())

    with writer.as_default():
      summary_ops_v2.trace_on(graph=True)

      with self.assertRaises(ValueError):
        summary_ops_v2.trace_export('trace', step=())

  @test_util.run_v2_only
  def test_scalar_summary_v2__family(self):
    # [FUNCTION DOC]
    # Test (TF2 only): version 1 scalar with a family argument forwards to the mocked version 2
    # scalar with the family applied as a name prefix, repeated around the name (the family
    # followed by the family-qualified scope), at step one; returns an empty string tensor.
    """Tests `family` arg handling when scalar v2 is invoked."""
    with test.mock.patch.object(
        tb_summary, 'scalar', autospec=True) as mock_scalar_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=1):
        tensor = summary_lib.scalar(
            'float', constant_op.constant(2.5), family='otter')
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_scalar_v2.assert_called_once_with(
        name='otter/otter/float',
        data=constant_op.constant(2.5),
        step=1,
    )

  @test_util.run_v2_only
  def test_scalar_summary_v2__family_w_outer_scope(self):
    # [FUNCTION DOC]
    # Test (TF2 only): same as the family test but inside an outer name scope; asserts the
    # forwarded name is the family, the outer scope, the family again and the base name.
    """Tests `family` arg handling when there is an outer scope."""
    with test.mock.patch.object(
        tb_summary, 'scalar', autospec=True) as mock_scalar_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=1):
        with ops.name_scope_v2('sea'):
          tensor = summary_lib.scalar(
              'float', constant_op.constant(3.5), family='crabnet')
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_scalar_v2.assert_called_once_with(
        name='crabnet/sea/crabnet/float',
        data=constant_op.constant(3.5),
        step=1,
    )

  @test_util.run_v2_only
  def test_scalar_summary_v2__v1_set_step(self):
    # [FUNCTION DOC]
    # Test (TF2 only): creates a version 1 global step and assigns 1024, then writes a scalar with
    # a default writer that has no explicit step. Asserts the mocked version 2 scalar receives
    # step 1024, i.e. the global step is used as fallback. Side effect: creates a global step in
    # the default graph.
    """Tests scalar v2 invocation when v1 step is set."""
    global_step = training_util.create_global_step()
    global_step.assign(1024)
    with test.mock.patch.object(
        tb_summary, 'scalar', autospec=True) as mock_scalar_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default():
        i = constant_op.constant(2.5)
        tensor = summary_lib.scalar('float', i)
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_scalar_v2.assert_called_once_with(name='float', data=i, step=1024)

  @test_util.run_v2_only
  def test_image_summary_v2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): version 1 image with a family, maximum outputs three, inside an outer scope
    # at step two forwards to the mocked version 2 image with the combined family/scope name, the
    # data, the step and max outputs; returns an empty string tensor.
    """Tests image v2 invocation."""
    with test.mock.patch.object(
        tb_summary, 'image', autospec=True) as mock_image_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=2):
        i = array_ops.ones((5, 4, 4, 3))
        with ops.name_scope_v2('outer'):
          tensor = summary_lib.image('image', i, max_outputs=3, family='family')
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_image_v2.assert_called_once_with(
        name='family/outer/family/image',
        data=i,
        step=2,
        max_outputs=3,
    )

  @test_util.run_v2_only
  def test_histogram_summary_v2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): version 1 histogram with a family at step three forwards to the mocked
    # version 2 histogram with the family-prefixed name, data and step; returns an empty string
    # tensor.
    """Tests histogram v2 invocation."""
    with test.mock.patch.object(
        tb_summary, 'histogram', autospec=True) as mock_histogram_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=3):
        i = array_ops.ones((1024,))
        tensor = summary_lib.histogram('histogram', i, family='family')
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_histogram_v2.assert_called_once_with(
        name='family/family/histogram',
        data=i,
        step=3,
    )

  @test_util.run_v2_only
  def test_audio_summary_v2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): version 1 audio inside an outer scope at step ten forwards to the mocked
    # version 2 audio with the scoped name, data, sample rate 0.2, step and max outputs three;
    # returns an empty string tensor.
    """Tests audio v2 invocation."""
    with test.mock.patch.object(
        tb_summary, 'audio', autospec=True) as mock_audio_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=10):
        i = array_ops.ones((5, 3, 4))
        with ops.name_scope_v2('dolphin'):
          tensor = summary_lib.audio('wave', i, 0.2, max_outputs=3)
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_audio_v2.assert_called_once_with(
        name='dolphin/wave',
        data=i,
        sample_rate=0.2,
        step=10,
        max_outputs=3,
    )

  @test_util.run_v2_only
  def test_audio_summary_v2__2d_tensor(self):
    # [FUNCTION DOC]
    # Test (TF2 only): version 1 audio given a two-dimensional tensor forwards to the mocked
    # version 2 audio with data reshaped to three dimensions by appending a channel axis of size
    # one; asserts the forwarded data equals an all-ones tensor of shape five, three, one.
    """Tests audio v2 invocation with 2-D tensor input."""
    with test.mock.patch.object(
        tb_summary, 'audio', autospec=True) as mock_audio_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=11):
        input_2d = array_ops.ones((5, 3))
        tensor = summary_lib.audio('wave', input_2d, 0.2, max_outputs=3)

    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)

    mock_audio_v2.assert_called_once_with(
        name='wave',
        data=test.mock.ANY,
        sample_rate=0.2,
        step=11,
        max_outputs=3,
    )
    input_3d = array_ops.ones((5, 3, 1))  # 3-D input tensor
    self.assertAllEqual(mock_audio_v2.call_args[1]['data'], input_3d)

  @test_util.run_v2_only
  def test_text_summary_v2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): version 1 text of a string constant at step twenty-two forwards to the
    # mocked version 2 text with the name, data and step; returns an empty string tensor.
    """Tests text v2 invocation."""
    with test.mock.patch.object(
        tb_summary, 'text', autospec=True) as mock_text_v2:
      with summary_ops_v2.create_summary_file_writer(
          self.get_temp_dir()).as_default(step=22):
        i = constant_op.constant('lorem ipsum', dtype=dtypes.string)
        tensor = summary_lib.text('text', i)
    # Returns empty string.
    self.assertEqual(tensor.numpy(), b'')
    self.assertEqual(tensor.dtype, dtypes.string)
    mock_text_v2.assert_called_once_with(name='text', data=i, step=22)


if __name__ == '__main__':
  test.main()
