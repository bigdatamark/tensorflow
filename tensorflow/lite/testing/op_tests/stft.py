# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the short-time Fourier
# transform. Defines and registers one generator function that creates parameterized graphs and
# passes them, with default extra conversion options, to the shared zip-test machinery.
# Import-time side effect: registers the generator in the global registry of test-making
# functions.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for stft."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraConvertOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_stft_tests(options):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it tests the two-dimensional real FFT, but the graph built
  # here applies the short-time Fourier transform.
  # Generates the STFT test set. The options argument is the test-run configuration passed to
  # the zip-test builder. Parameters sweep float32 input of shapes 1-D, 2-D and 3-D, frame
  # length (4 or 8), frame step (1, 2 or 4) and FFT length (unspecified, 2, 4 or 8). A default
  # extra-conversion-options object is passed to the builder. Returns nothing.
  """Make a set of tests to do rfft2d."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[8], [8, 16], [3, 1, 4]],
      "frame_length": [4, 8],
      "frame_step": [1, 2, 4],
      "fft_length": [None, 2, 4, 8],
  }]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Builds the graph for one parameter combination: a placeholder named input of the given
    # type and shape, followed by the signal-processing short-time Fourier transform with the
    # given frame length, frame step and FFT length. Returns a pair: list with the input
    # placeholder and list with the transform output.
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    outs = tf.signal.stft(
        input_value,
        frame_length=parameters["frame_length"],
        frame_step=parameters["frame_step"],
        fft_length=parameters["fft_length"])
    return [input_value], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Generates input data and reference outputs: random data of the parameter type and shape is
    # fed into the graph in the given session. Returns a pair: list with the input array and the
    # evaluated outputs.
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  extra_convert_options = ExtraConvertOptions()
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_convert_options)
