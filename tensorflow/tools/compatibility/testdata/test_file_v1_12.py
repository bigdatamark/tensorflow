# [MODULE DOC]
# Test data for the TF 1.x to 2.0 upgrade script: a test module written against the compat.v1 API
# (version 1.12 era) that exercises renamed or re-argumented APIs (acos, rsqrt, serialize_sparse,
# serialize_many_sparse, argmax/argmin with the old dimension keyword, softmax cross entropy).
# It is executed by the upgrader tests both before and after conversion; it defines one test class
# and runs the TensorFlow test main when invoked as a script. No import-time side effects other
# than imports and a version constant.
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
"""Tests for tf upgrader."""

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib

_TEST_VERSION = 1


class TestUpgrade(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case holding the old-API usages that the upgrader must be able to convert. Subclasses the
  # TensorFlow test base class; setUpClass records which API version is loaded and setUp enables
  # v2 behavior. Each test method checks numeric results of one family of renamed APIs.
  """Test various APIs that have been changed in 2.0."""

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture: stores on the class the API version, 1 if the loaded module still has
    # the contrib attribute, otherwise 2. Takes the class, returns nothing.
    cls._tf_api_version = 1 if hasattr(tf, 'contrib') else 2

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture: turns on v2 behavior through the compat.v1 entry point (eager execution
    # and v2 semantics). Takes no inputs, returns nothing; changes global TensorFlow behavior.
    tf.compat.v1.enable_v2_behavior()

  def testRenames(self):
    # [FUNCTION DOC]
    # Test: checks the old-name acos of 0.5 is about 1.0472 and rsqrt of 4.0 is 0.5, using
    # approximate comparison. Fails with an assertion error on mismatch.
    self.assertAllClose(1.04719755, tf.acos(0.5))
    self.assertAllClose(0.5, tf.rsqrt(4.0))

  def testSerializeSparseTensor(self):
    # [FUNCTION DOC]
    # Test: builds a sparse tensor with one index and one value, serializes it with the old
    # serialize_sparse API as a string tensor, and asserts the result has shape of three entries
    # and that its first element is non-empty.
    sp_input = tf.SparseTensor(
        indices=tf.constant([[1]], dtype=tf.int64),
        values=tf.constant([2], dtype=tf.int64),
        dense_shape=[2])

    with self.cached_session():
      serialized_sp = tf.serialize_sparse(sp_input, 'serialize_name', tf.string)
      self.assertEqual((3,), serialized_sp.shape)
      self.assertTrue(serialized_sp[0].numpy())  # check non-empty

  def testSerializeManySparse(self):
    # [FUNCTION DOC]
    # Test: builds a batch-of-one sparse tensor, serializes it with the old serialize_many_sparse
    # API, and asserts the serialized result has shape one by three.
    sp_input = tf.SparseTensor(
        indices=tf.constant([[0, 1]], dtype=tf.int64),
        values=tf.constant([2], dtype=tf.int64),
        dense_shape=[1, 2])

    with self.cached_session():
      serialized_sp = tf.serialize_many_sparse(
          sp_input, 'serialize_name', tf.string)
      self.assertEqual((1, 3), serialized_sp.shape)

  def testArgMaxMin(self):
    # [FUNCTION DOC]
    # Test: uses the old dimension keyword of argmax and argmin (renamed to axis in 2.0) on a
    # one-row matrix and asserts the resulting indices: argmax along dimension 1 gives 1, along
    # dimension 0 gives zeros, argmin along dimension 1 gives 0.
    self.assertAllClose(
        [1],
        tf.argmax([[1, 3, 2]], name='abc', dimension=1))
    self.assertAllClose(
        [0, 0, 0],
        tf.argmax([[1, 3, 2]], dimension=0))
    self.assertAllClose(
        [0],
        tf.argmin([[1, 3, 2]], name='abc', dimension=1))

  def testSoftmaxCrossEntropyWithLogits(self):
    # [FUNCTION DOC]
    # Test: calls both softmax cross entropy with logits and its v2 variant with the same logits
    # and one-hot-like labels and asserts both yield about 0.40318608.
    out = tf.nn.softmax_cross_entropy_with_logits(
        logits=[0.1, 0.8], labels=[0, 1])
    self.assertAllClose(out, 0.40318608)
    out = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=[0.1, 0.8], labels=[0, 1])
    self.assertAllClose(out, 0.40318608)


if __name__ == "__main__":
  test_lib.main()
