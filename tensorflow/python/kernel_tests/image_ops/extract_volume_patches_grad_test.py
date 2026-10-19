# [MODULE DOC]
# Gradient tests for the extract-volume-patches op (the three-dimensional analogue of image
# patch extraction). Contains a numeric gradient check over four parameterized shape, kernel and
# stride combinations with both padding modes, and a regression test that building the gradient
# for a moderately large volume is fast. No import-time side effects besides the test main entry
# point when run as a script.
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
"""Tests for ExtractVolumePatches gradient."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed as random_seed_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ExtractVolumePatchesGradTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the gradient of the extract-volume-patches op. Has no shared
  # fixtures; each test method creates its own random input and compares gradients.
  """Gradient-checking for ExtractVolumePatches op."""

  @parameterized.parameters([
      {
          'in_shape': [2, 5, 5, 5, 3],
          'ksizes': [1, 1, 1, 1, 1],
          'strides': [1, 2, 3, 4, 1],
      },
      {
          'in_shape': [2, 7, 7, 7, 3],
          'ksizes': [1, 3, 3, 3, 1],
          'strides': [1, 1, 1, 1, 1],
      },
      {
          'in_shape': [2, 5, 7, 6, 3],
          'ksizes': [1, 3, 2, 2, 1],
          'strides': [1, 1, 1, 1, 1],
      },
      {
          'in_shape': [2, 7, 8, 6, 3],
          'ksizes': [1, 2, 3, 2, 1],
          'strides': [1, 2, 4, 3, 1],
      },
  ])
  def testGradient(self, in_shape, ksizes, strides):
    # [FUNCTION DOC]
    # Test, parameterized over input shape (batch, planes, rows, columns, depth), five-element
    # patch sizes and five-element strides. Skipped when a GPU is available (known bug
    # reference in the skip message). Seeds the graph and numpy random generators, builds a
    # random float input, and for both padding modes compares theoretical and numeric Jacobians
    # of the patch extraction, printing the maximum error and asserting it is below 0.0001.
    if test_util.is_gpu_available():
      self.skipTest('b/171837334: skip gpu test.')

    # Set graph seed for determinism.
    random_seed = 42
    random_seed_lib.set_random_seed(random_seed)

    with self.cached_session():
      np.random.seed(random_seed)
      input_val = constant_op.constant(
          np.random.random(in_shape), dtype=dtypes.float32)

      for padding in ['VALID', 'SAME']:

        def extract(in_val, ksizes=ksizes, strides=strides, padding=padding):
          # [FUNCTION DOC]
          # HELPER: function under differentiation for the gradient check. Input: in_val is the
          # volume tensor; the other parameters are defaults bound from the enclosing scope (patch
          # sizes, strides, padding mode). Returns the output of the extract-volume-patches op.
          return array_ops.extract_volume_patches(in_val, ksizes, strides,
                                                  padding)

        rtn = gradient_checker_v2.compute_gradient(extract, [input_val])
        err = gradient_checker_v2.max_error(*rtn)

        print('extract_volume_patches gradient err: %.4e' % err)
        self.assertLess(err, 1e-4)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testConstructGradientWithLargeVolumes(self, use_tape):
    # [FUNCTION DOC]
    # Regression test for a reported slow gradient construction. Parameterized on whether a real
    # gradient tape is used (always true, plus true again only when eager, via a set). Builds a
    # variable of shape 4, 8, 32, 32, 1, extracts five by five by five patches with same padding,
    # builds the gradient with respect to the variable and asserts it is not null.
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      batch_size = 4
      planes = 8
      height = 32
      width = 32
      ksize = 5
      shape = (batch_size, planes, height, width, 1)

      volumes = variables.Variable(
          np.random.uniform(size=np.prod(shape)).reshape(shape), name='inputs')

      tape.watch(volumes)
      patches = array_ops.extract_volume_patches(
          volumes,
          ksizes=[1, ksize, ksize, ksize, 1],
          strides=[1, 1, 1, 1, 1],
          padding='SAME')
      # Github issue: #20146
      # tf.extract_volume_patches() gradient very slow at graph construction
      # time.
      gradients = tape.gradient(patches, volumes)
      # Won't time out.
      self.assertIsNotNone(gradients)


if __name__ == '__main__':
  test.main()
