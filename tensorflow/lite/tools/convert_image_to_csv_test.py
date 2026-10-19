# [MODULE DOC]
# Unit tests for the image-to-CSV conversion tool: checks image loading size, grayscale and color
# conversion, value ranges for JPEG and PNG test data, missing files and CSV formatting.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests image file conversion utilities."""
import os

import numpy as np

from tensorflow.lite.tools import convert_image_to_csv
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test

PREFIX_PATH = resource_loader.get_path_to_datafile("../../core/lib/")


class ConvertImageToCsvTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the get_image and array_to_int_csv functions, using image test data files from the
  # core library test data directory (path held in a module-level prefix).

  def testGetImageRaisesMissingFile(self):
    # [FUNCTION DOC]
    # Scenario: loads a nonexistent JPEG path.
    # Asserts a not-found error is raised.
    image_path = os.path.join(PREFIX_PATH, "jpeg", "testdata", "no_such.jpg")
    with self.assertRaises(NotFoundError):
      _ = convert_image_to_csv.get_image(64, 96, False, image_path)

  def testGetImageSizeIsCorrect(self):
    # [FUNCTION DOC]
    # Scenario: loads a small JPEG at width 64, height 96 in color.
    # Asserts the shape is height 96, width 64, 3 channels.
    image_path = os.path.join(PREFIX_PATH, "jpeg", "testdata", "small.jpg")
    image_data = convert_image_to_csv.get_image(64, 96, False, image_path)
    self.assertEqual((96, 64, 3), image_data.shape)

  def testGetImageConvertsToGrayscale(self):
    # [FUNCTION DOC]
    # Scenario: loads a medium JPEG at width 40, height 20 with grayscale on.
    # Asserts the shape is 20 by 40 with one channel.
    image_path = os.path.join(PREFIX_PATH, "jpeg", "testdata", "medium.jpg")
    image_data = convert_image_to_csv.get_image(40, 20, True, image_path)
    self.assertEqual((20, 40, 1), image_data.shape)

  def testGetImageCanLoadPng(self):
    # [FUNCTION DOC]
    # Scenario: loads an RGBA PNG at 10 by 10 in color.
    # Asserts the shape is 10 by 10 with three channels (alpha dropped).
    image_path = os.path.join(PREFIX_PATH, "png", "testdata", "lena_rgba.png")
    image_data = convert_image_to_csv.get_image(10, 10, False, image_path)
    self.assertEqual((10, 10, 3), image_data.shape)

  def testGetImageConvertsGrayscaleToColor(self):
    # [FUNCTION DOC]
    # Scenario: loads a grayscale PNG at width 23, height 19 requesting color.
    # Asserts the shape is 19 by 23 with three channels.
    image_path = os.path.join(PREFIX_PATH, "png", "testdata", "lena_gray.png")
    image_data = convert_image_to_csv.get_image(23, 19, False, image_path)
    self.assertEqual((19, 23, 3), image_data.shape)

  def testGetImageColorValuesInRange(self):
    # [FUNCTION DOC]
    # Scenario: loads a small JPEG in color at width 47, height 31.
    # Asserts all values lie between 0 and 255.
    image_path = os.path.join(PREFIX_PATH, "jpeg", "testdata", "small.jpg")
    image_data = convert_image_to_csv.get_image(47, 31, False, image_path)
    self.assertLessEqual(0, np.min(image_data))
    self.assertGreaterEqual(255, np.max(image_data))

  def testGetImageGrayscaleValuesInRange(self):
    # [FUNCTION DOC]
    # Scenario: loads a small JPEG as grayscale at width 27, height 33.
    # Asserts all values lie between 0 and 255.
    image_path = os.path.join(PREFIX_PATH, "jpeg", "testdata", "small.jpg")
    image_data = convert_image_to_csv.get_image(27, 33, True, image_path)
    self.assertLessEqual(0, np.min(image_data))
    self.assertGreaterEqual(255, np.max(image_data))

  def testArrayToIntCsv(self):
    # [FUNCTION DOC]
    # Scenario: converts a two by two integer array.
    # Asserts the result is the string of values 1 to 4 separated by commas.
    csv_string = convert_image_to_csv.array_to_int_csv(
        np.array([[1, 2], [3, 4]]))
    self.assertEqual("1,2,3,4", csv_string)

  def testArrayToIntCsvRounding(self):
    # [FUNCTION DOC]
    # Scenario: converts a two by two array of floating point values with zero fraction.
    # Asserts the output has integer text, 1 to 4 separated by commas.
    csv_string = convert_image_to_csv.array_to_int_csv(
        np.array([[1.0, 2.0], [3.0, 4.0]]))
    self.assertEqual("1,2,3,4", csv_string)


if __name__ == "__main__":
  test.main()
