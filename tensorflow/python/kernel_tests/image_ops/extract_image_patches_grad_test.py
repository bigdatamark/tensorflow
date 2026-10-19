# [MODULE DOC]
# Gradient tests for the extract-image-patches op. Numerically checks the gradient against
# finite differences for a table of input shapes, patch sizes, strides and dilation rates with
# both padding modes, with fully known and partially unknown static shapes, and adds regression
# tests that building the gradient for large images is fast (and does not run out of memory) and
# that the backward pass works under XLA compilation. No import-time side effects besides the
# test main entry point when run as a script.
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
"""Tests for ExtractImagePatches gradient."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as random_seed_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ExtractImagePatchesGradTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the extract-image-patches gradient. Holds a class-level table of
  # seven scenarios (input shape, patch sizes, strides, rates) used by the numeric gradient
  # checks, plus a helper that varies which dimensions are statically unknown.
  """Gradient-checking for ExtractImagePatches op."""

  _TEST_CASES = [
      {
          'in_shape': [2, 5, 5, 3],
          'ksizes': [1, 1, 1, 1],
          'strides': [1, 2, 3, 1],
          'rates': [1, 1, 1, 1],
      },
      {
          'in_shape': [2, 7, 7, 3],
          'ksizes': [1, 3, 3, 1],
          'strides': [1, 1, 1, 1],
          'rates': [1, 1, 1, 1],
      },
      {
          'in_shape': [2, 8, 7, 3],
          'ksizes': [1, 2, 2, 1],
          'strides': [1, 1, 1, 1],
          'rates': [1, 1, 1, 1],
      },
      {
          'in_shape': [2, 7, 8, 3],
          'ksizes': [1, 3, 2, 1],
          'strides': [1, 4, 3, 1],
          'rates': [1, 1, 1, 1],
      },
      {
          'in_shape': [1, 15, 20, 3],
          'ksizes': [1, 4, 3, 1],
          'strides': [1, 1, 1, 1],
          'rates': [1, 2, 4, 1],
      },
      {
          'in_shape': [2, 7, 8, 1],
          'ksizes': [1, 3, 2, 1],
          'strides': [1, 3, 2, 1],
          'rates': [1, 2, 2, 1],
      },
      {
          'in_shape': [2, 8, 9, 4],
          'ksizes': [1, 2, 2, 1],
          'strides': [1, 4, 2, 1],
          'rates': [1, 3, 2, 1],
      },
  ]

  def testGradient(self):
    # [FUNCTION DOC]
    # Test: seeds the graph and numpy random generators for determinism, then for each scenario in
    # the table and both padding modes computes the theoretical and numeric Jacobians of the patch
    # extraction with respect to a random float input and asserts the maximum error is below
    # 0.0001.
    # Set graph seed for determinism.
    random_seed = 42
    random_seed_lib.set_random_seed(random_seed)

    with self.cached_session():
      for test_case in self._TEST_CASES:
        np.random.seed(random_seed)
        in_shape = test_case['in_shape']
        in_val = constant_op.constant(
            np.random.random(in_shape), dtype=dtypes.float32)
        # Avoid `dangerous-default-value` pylint error by creating default
        # args to `extract` as tuples.
        ksizes = tuple(test_case['ksizes'])
        strides = tuple(test_case['strides'])
        rates = tuple(test_case['rates'])

        for padding in ['VALID', 'SAME']:

          def extract(in_val,
                      ksizes=ksizes,
                      strides=strides,
                      rates=rates,
                      padding=padding):
            # [FUNCTION DOC]
            # HELPER: function under differentiation for the gradient check. Input: in_val is the
            # image tensor; the remaining parameters are defaults bound from the current loop
            # iteration (patch sizes, strides, rates, padding mode) so each closure is
            # independent. Returns the patches tensor produced by the extract-image-patches op.
            return array_ops.extract_image_patches(in_val, ksizes, strides,
                                                   rates, padding)

          err = gradient_checker_v2.max_error(
              *gradient_checker_v2.compute_gradient(extract, [in_val]))
          self.assertLess(err, 1e-4)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testConstructGradientWithLargeImages(self, use_tape):
    # [FUNCTION DOC]
    # Regression test for a reported slowdown at gradient construction time. Parameterized over
    # whether a real gradient tape is used (always true, plus true again only if the context is
    # eager, via a set that removes duplicates). Builds a four by 512 by 512 single channel image
    # variable, extracts five by five patches with same padding, builds the gradient, and asserts
    # that it is not null (the test would time out if construction were slow).
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      batch_size = 4
      # Prevent OOM by setting reasonably large image size (b/171808681).
      height = 512
      width = 512
      ksize = 5
      shape = (batch_size, height, width, 1)
      images = variables.Variable(
          np.random.uniform(size=np.prod(shape)).reshape(shape), name='inputs')
      tape.watch(images)
      patches = array_ops.extract_image_patches(images,
                                                ksizes=[1, ksize, ksize, 1],
                                                strides=[1, 1, 1, 1],
                                                rates=[1, 1, 1, 1],
                                                padding='SAME')
      # Github issue: #20146
      # tf.image.extract_image_patches() gradient very slow at graph
      # construction time.
      gradients = tape.gradient(patches, images)
      # Won't time out.
      self.assertIsNotNone(gradients)

  def _VariableShapeGradient(self, test_shape_pattern):
    # [FUNCTION DOC]
    # HELPER: shared body of the four shape-pattern gradient tests.
    # KNOWN ISSUE: the existing docstring is a truncated sentence (it breaks after "of" and
    # continues on a separate line) and does not describe the behavior. Also the pattern's
    # non-null entries are only used as a marker: any non-null entry keeps the true dimension
    # size, the numeric value itself is never used as a size.
    # Input: test_shape_pattern is a list of four entries, one per dimension; a null entry makes
    # that dimension statically unknown in the placeholder, any other entry keeps the real size
    # from the scenario table. In graph mode, for each scenario and both padding modes, feeds a
    # random array into a placeholder of that partially unknown shape, evaluates the patches to
    # learn the output shape, computes the numeric gradient error and asserts it is below 0.0001.
    # Returns nothing; uses graph-seeded and numpy-seeded randomness for determinism.
    """Use test_shape_pattern to infer which dimensions are of

    variable size.
    """
    # Testing shape gradient requires graph mode.
    with ops.Graph().as_default():
      # Set graph seed for determinism.
      random_seed = 42
      random_seed_lib.set_random_seed(random_seed)

      with self.test_session():
        for test_case in self._TEST_CASES:
          np.random.seed(random_seed)
          in_shape = test_case['in_shape']
          test_shape = [
              x if x is None else y
              for x, y in zip(test_shape_pattern, in_shape)
          ]
          in_val = array_ops.placeholder(shape=test_shape, dtype=dtypes.float32)

          feed_dict = {in_val: np.random.random(in_shape)}
          for padding in ['VALID', 'SAME']:
            out_val = array_ops.extract_image_patches(in_val,
                                                      test_case['ksizes'],
                                                      test_case['strides'],
                                                      test_case['rates'],
                                                      padding)
            out_val_tmp = out_val.eval(feed_dict=feed_dict)
            out_shape = out_val_tmp.shape

            err = gradient_checker.compute_gradient_error(
                in_val, in_shape, out_val, out_shape)
            self.assertLess(err, 1e-4)

  def test_BxxC_Gradient(self):
    # [FUNCTION DOC]
    # Test: gradient check where batch and channel dimensions are known and height and width are
    # unknown at graph construction time.
    self._VariableShapeGradient([-1, None, None, -1])

  def test_xHWx_Gradient(self):
    # [FUNCTION DOC]
    # Test: gradient check where height and width are known and batch and channel dimensions are
    # unknown at graph construction time.
    self._VariableShapeGradient([None, -1, -1, None])

  def test_BHWC_Gradient(self):
    # [FUNCTION DOC]
    # Test: gradient check where all four dimensions are marked as known.
    self._VariableShapeGradient([-1, -1, -1, -1])

  def test_AllNone_Gradient(self):
    # [FUNCTION DOC]
    # Test: gradient check where all four dimensions are statically unknown.
    self._VariableShapeGradient([None, None, None, None])

  def testJitCompile(self):
    # [FUNCTION DOC]
    # Regression test for backward pass failing under XLA compilation. Wraps the patch
    # extraction in a function compiled with just-in-time compilation, applies it to a double
    # precision 4 by 512 by 512 by 1 variable with five by five patches and same padding under a
    # gradient tape, and asserts the computed gradient is not null.
    with test_util.AbstractGradientTape(use_tape=True) as tape:
      shape = (4, 512, 512, 1)
      ksize = 5

      images = variables.Variable(np.random.uniform(size=shape), name='inputs')
      tape.watch(images)

      # Github issues: #59058, #59061
      # tf.image.extract_image_patches() does not support backward pass
      # when compiled with XLA.
      extract_image_patches_jit = polymorphic_function.function(
          array_ops.extract_image_patches, jit_compile=True
      )
      patches = extract_image_patches_jit(
          images,
          ksizes=[1, ksize, ksize, 1],
          strides=[1] * 4,
          rates=[1] * 4,
          padding='SAME',
      )

      gradients = tape.gradient(patches, images)
      self.assertIsNotNone(gradients)


if __name__ == '__main__':
  test.main()
