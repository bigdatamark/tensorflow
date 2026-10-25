# [MODULE DOC]
# Test module for image op gradients. It contains no test logic of its own: it aliases the
# concrete test case classes defined in the shared base module (nearest-neighbor, bilinear and
# bicubic resize, scale-and-translate, crop-and-resize and RGB-to-HSV gradient tests) under
# public names so the test runner discovers them, and runs the test main when executed directly.
# Importing it only binds those aliases.
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
"""Functional tests for Image Op Gradients."""

from tensorflow.python.ops import image_grad_test_base as test_base
from tensorflow.python.platform import test

ResizeNearestNeighborOpTest = test_base.ResizeNearestNeighborOpTestBase
ResizeBilinearOpTest = test_base.ResizeBilinearOpTestBase
ResizeBicubicOpTest = test_base.ResizeBicubicOpTestBase
ScaleAndTranslateOpTest = test_base.ScaleAndTranslateOpTestBase
CropAndResizeOpTest = test_base.CropAndResizeOpTestBase
RGBToHSVOpTest = test_base.RGBToHSVOpTestBase

if __name__ == "__main__":
  test.main()
