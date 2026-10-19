# [MODULE DOC]
# Benchmark (not a correctness test) for the JPEG decoding op. Defines a benchmark class that
# times decoding of small and medium sample JPEG files (and a tiled, enlarged version of the
# medium one) at parallelism of 1 and 100, with three variants: plain decode, decode followed by
# a separate crop, and the fused decode-and-crop op. Results are reported through the benchmark
# reporting facility. The sample images are located through the resource loader inside the core
# library test data. Nothing happens at import time other than the benchmark runner entry point
# when the file is run as a script.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DecodeJpegOp."""

import os
import time

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class DecodeJpegBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class measuring wall time of the JPEG decode op for several image sizes, crop
  # strategies and degrees of parallelism. One private helper performs a single timed
  # measurement; three benchmark methods (small, medium, large) call it for each variant and
  # report the durations. It holds no state of its own.
  """Evaluate tensorflow DecodeJpegOp performance."""

  def _evalDecodeJpeg(self,
                      image_name,
                      parallelism,
                      num_iters,
                      crop_during_decode=None,
                      crop_window=None,
                      tile=None):
    # [FUNCTION DOC]
    # HELPER: timing routine shared by the benchmark methods of this class.
    # Inputs: image_name is the sample file name inside the core JPEG test data directory;
    # parallelism is how many independent decode ops are grouped and run together; num_iters is
    # the number of timed runs; crop_during_decode (optional) selects the fused decode-and-crop
    # op over separate decode and crop ops and is ignored when crop_window is absent; crop_window
    # (optional) is offset height, offset width, target height, target width; tile (optional) is
    # a per-dimension repeat list used to build a larger synthetic image by tiling the decoded
    # sample and re-encoding it as JPEG.
    # Behavior: resets the default graph, locates the file (trying a second relative path when
    # the first does not exist), stores the encoded image bytes in a variable, builds the decode
    # ops, initializes variables in a new session, runs three untimed warm-up iterations, then
    # times the requested number of runs. Returns the elapsed wall time in seconds. Side effects:
    # resets global graph state, reads files and creates a session.
    """Evaluate DecodeJpegOp for the given image.

    TODO(tanmingxing): add decoding+cropping as well.

    Args:
      image_name: a string of image file name (without suffix).
      parallelism: the number of concurrent decode_jpeg ops to be run.
      num_iters: number of iterations for evaluation.
      crop_during_decode: If true, use fused DecodeAndCropJpeg instead of
          separate decode and crop ops. It is ignored if crop_window is None.
      crop_window: if not None, crop the decoded image. Depending on
          crop_during_decode, cropping could happen during or after decoding.
      tile: if not None, tile the image to composite a larger fake image.

    Returns:
      The duration of the run in seconds.
    """
    ops.reset_default_graph()

    image_file_path = resource_loader.get_path_to_datafile(
        os.path.join('core', 'lib', 'jpeg', 'testdata', image_name))

    # resource_loader does not seem to work well under benchmark runners.
    # So if the above path is not available, try another way to access the file:
    if not os.path.exists(image_file_path):
      image_file_path = resource_loader.get_path_to_datafile(
          os.path.join(
              '..', '..', 'core', 'lib', 'jpeg', 'testdata', image_name))

    if tile is None:
      image_content = variable_scope.get_variable(
          'image_%s' % image_name,
          initializer=io_ops.read_file(image_file_path))
    else:
      single_image = image_ops.decode_jpeg(
          io_ops.read_file(image_file_path), channels=3, name='single_image')
      # Tile the image to composite a new larger image.
      tiled_image = array_ops.tile(single_image, tile)
      image_content = variable_scope.get_variable(
          'tiled_image_%s' % image_name,
          initializer=image_ops.encode_jpeg(tiled_image))

    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      images = []
      for _ in range(parallelism):
        if crop_window is None:
          # No crop.
          image = image_ops.decode_jpeg(image_content, channels=3)
        elif crop_during_decode:
          # combined decode and crop.
          image = image_ops.decode_and_crop_jpeg(
              image_content, crop_window, channels=3)
        else:
          # separate decode and crop.
          image = image_ops.decode_jpeg(image_content, channels=3)
          image = image_ops.crop_to_bounding_box(
              image,
              offset_height=crop_window[0],
              offset_width=crop_window[1],
              target_height=crop_window[2],
              target_width=crop_window[3])

        images.append(image)
      r = control_flow_ops.group(*images)

      for _ in range(3):
        # Skip warm up time.
        self.evaluate(r)

      start_time = time.time()
      for _ in range(num_iters):
        self.evaluate(r)
      end_time = time.time()
    return end_time - start_time

  def benchmarkDecodeJpegSmall(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable names are swapped relative to what is measured. The value named
    # decode-crop is measured with the fused option turned off (separate decode then crop) and
    # the value named decode-after-crop is measured with the fused decode-and-crop option on, so
    # the reported benchmark names do not match the strategies. The docstring says "single
    # DecodeImageOp" although the JPEG op is timed.
    # Benchmark: for parallelism 1 and 100 and ten iterations, times plain decode, separate
    # decode plus crop, and fused decode-and-crop of the small sample image with a 50 by 50 crop
    # window at offset 10, 10, and reports three wall times per parallelism. Returns nothing.
    """Evaluate single DecodeImageOp for small size image."""
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    for parallelism in [1, 100]:
      duration_decode = self._evalDecodeJpeg('small.jpg', parallelism,
                                             num_iters)
      duration_decode_crop = self._evalDecodeJpeg('small.jpg', parallelism,
                                                  num_iters, False, crop_window)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'small.jpg', parallelism, num_iters, True, crop_window)
      self.report_benchmark(
          name='decode_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)

  def benchmarkDecodeJpegMedium(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same swapped naming as the small benchmark: the "decode_crop" report is the
    # separate decode-then-crop case and the "decode_after_crop" report is the fused
    # decode-and-crop case. The docstring also says DecodeImageOp instead of the JPEG op.
    # Benchmark: same three variants as the small-image benchmark, run on the medium sample
    # image, for parallelism 1 and 100 with ten iterations; reports three wall times each.
    # Returns nothing.
    """Evaluate single DecodeImageOp for medium size image."""
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    for parallelism in [1, 100]:
      duration_decode = self._evalDecodeJpeg('medium.jpg', parallelism,
                                             num_iters)
      duration_decode_crop = self._evalDecodeJpeg('medium.jpg', parallelism,
                                                  num_iters, False, crop_window)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, True, crop_window)
      self.report_benchmark(
          name='decode_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)

  def benchmarkDecodeJpegLarge(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the plain-decode call the tile list is passed as the fourth positional
    # argument, which is the fused-crop flag, not the tile parameter, so that measurement is
    # not tiled (it is just a plain decode of the medium image). The crop variants pass the tile
    # correctly. Reported names are also swapped as in the other benchmarks (decode_crop is the
    # separate decode-then-crop case, decode_after_crop is the fused case).
    # Benchmark: intended to time decoding of a large image made by tiling the medium sample four
    # by four, at parallelism 1 and 100 with ten iterations, in the three variants; reports three
    # wall times each. Returns nothing.
    """Evaluate single DecodeImageOp for large size image."""
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    tile = [4, 4, 1]
    for parallelism in [1, 100]:
      # Tile the medium size image to composite a larger fake image.
      duration_decode = self._evalDecodeJpeg('medium.jpg', parallelism,
                                             num_iters, tile)
      duration_decode_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, False, crop_window, tile)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, True, crop_window, tile)
      self.report_benchmark(
          name='decode_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)


if __name__ == '__main__':
  test.main()
