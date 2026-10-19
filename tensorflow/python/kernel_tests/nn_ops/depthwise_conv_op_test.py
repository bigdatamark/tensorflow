# [MODULE DOC]
# Test entry module for the depthwise 2-D convolution op tests. It contains no test logic of its
# own: it exposes the shared base test case from the sibling base module under the name
# DepthwiseConv2DTest so that the test runner discovers and executes every test method defined
# there (value checks against a reference implementation, gradient checks, and CPU-versus-GPU
# comparisons) with the default, non-deterministic op settings.
# When run as a script it starts the standard TensorFlow test runner. Importing it has no side
# effects other than importing the base module and the test platform module.
# Determinism tests live in a separate file so the cuDNN algorithm cache is not shared.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for depthwise convolutional operations."""

from tensorflow.python.kernel_tests.nn_ops import depthwise_conv_op_base
from tensorflow.python.platform import test

DepthwiseConv2DTest = depthwise_conv_op_base.DepthwiseConv2DBase

if __name__ == "__main__":
  test.main()
