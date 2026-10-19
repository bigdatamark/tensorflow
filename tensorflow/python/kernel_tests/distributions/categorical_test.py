# [MODULE DOC]
# Unit tests for the Categorical distribution in the distributions package of TensorFlow ops.
# Covers construction from probabilities or logits, static and dynamic shapes and event size,
# dtypes, sampling with unknown shapes, the probability mass function (with batch and broadcasting),
# the cumulative distribution function (static, dynamic and unknown-rank inputs, broadcasting
# against a Normal distribution), log-pmf shapes, entropy and its gradient, sampling statistics and
# sample shapes, non-reparameterization, mode, and the registered Categorical-to-Categorical KL
# divergence. Defines one module-level factory helper and one test class combining the TensorFlow
# test case with parameterized tests; running the file as a script runs the tests.
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
"""Tests for Categorical distribution."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import normal
from tensorflow.python.platform import test


def make_categorical(batch_shape, num_classes, dtype=dtypes.int32):
  # [FUNCTION DOC]
  # HELPER: factory producing random Categorical distributions for the shape and dtype tests.
  # Takes batch_shape (sequence of sizes), num_classes (an integer or a scalar integer tensor) and
  # an optional sample dtype (default 32-bit integer). Draws uniform random logits between minus 10
  # and 10 with shape batch_shape followed by num_classes, shifts them by minus 50, and returns a
  # Categorical built from those logits with the given dtype. Uses TensorFlow random ops.
  logits = random_ops.random_uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=dtypes.float32) - 50.
  return categorical.Categorical(logits, dtype=dtype)


class CategoricalTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the Categorical distribution; mixes the TensorFlow test case with absl
  # parameterized test support (used by the unknown-rank cdf test). Most tests are graph-mode only
  # (deprecated v1 decorator) and evaluate tensors inside a cached session; a few run in the
  # default mode. Logits are generally log probabilities shifted by minus 50 to check
  # shift invariance. No setUp or shared fixtures.

  @test_util.run_deprecated_v1
  def testP(self):
    # [FUNCTION DOC]
    # Scenario: Categorical from probabilities 0.2 and 0.8 (graph mode).
    # Expected: the probs property equals the input and the logits property has static shape two.
    p = [0.2, 0.8]
    dist = categorical.Categorical(probs=p)
    with self.cached_session():
      self.assertAllClose(p, dist.probs)
      self.assertAllEqual([2], dist.logits.get_shape())

  @test_util.run_deprecated_v1
  def testLogits(self):
    # [FUNCTION DOC]
    # Scenario: logits are the log of probabilities 0.2 and 0.8 minus 50 (graph mode).
    # Expected: probs and logits both have static shape two, probs equal the normalized probabilities
    # and logits equal the supplied logits.
    p = np.array([0.2, 0.8], dtype=np.float32)
    logits = np.log(p) - 50.
    dist = categorical.Categorical(logits=logits)
    with self.cached_session():
      self.assertAllEqual([2], dist.probs.get_shape())
      self.assertAllEqual([2], dist.logits.get_shape())
      self.assertAllClose(dist.probs, p)
      self.assertAllClose(dist.logits, logits)

  @test_util.run_deprecated_v1
  def testShapes(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): distributions over 10 classes for batch shapes scalar, one element and
    # 2x3x4, first with a Python integer class count and then with a constant tensor class count.
    # Expected: batch shape (static and dynamic) equals the requested shape, event shape is empty,
    # the event size evaluates to 10 and, for the static class count, is also available as a constant
    # value at graph build time.
    with self.cached_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_categorical(batch_shape, 10)
        self.assertAllEqual(batch_shape, dist.batch_shape)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor())
        self.assertAllEqual([], dist.event_shape)
        self.assertAllEqual([], dist.event_shape_tensor())
        self.assertEqual(10, dist.event_size.eval())
        # event_size is available as a constant because the shape is
        # known at graph build time.
        self.assertEqual(10, tensor_util.constant_value(dist.event_size))

      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_categorical(
            batch_shape, constant_op.constant(
                10, dtype=dtypes.int32))
        self.assertAllEqual(len(batch_shape), dist.batch_shape.ndims)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor())
        self.assertAllEqual([], dist.event_shape)
        self.assertAllEqual([], dist.event_shape_tensor())
        self.assertEqual(10, dist.event_size.eval())

  def testDtype(self):
    # [FUNCTION DOC]
    # Scenario: distributions over five classes with sample dtypes 32-bit int, 64-bit int and then
    # float16, float32 and float64.
    # Expected: dist dtype is the requested type, and sample and mode have it; probs and logits are
    # float32; entropy, prob and log_prob (with a 64-bit integer event) have the logits dtype; for each
    # float sample dtype the sample has that dtype.
    dist = make_categorical([], 5, dtype=dtypes.int32)
    self.assertEqual(dist.dtype, dtypes.int32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    dist = make_categorical([], 5, dtype=dtypes.int64)
    self.assertEqual(dist.dtype, dtypes.int64)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.probs.dtype, dtypes.float32)
    self.assertEqual(dist.logits.dtype, dtypes.float32)
    self.assertEqual(dist.logits.dtype, dist.entropy().dtype)
    self.assertEqual(
        dist.logits.dtype, dist.prob(np.array(
            0, dtype=np.int64)).dtype)
    self.assertEqual(
        dist.logits.dtype, dist.log_prob(np.array(
            0, dtype=np.int64)).dtype)
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      dist = make_categorical([], 5, dtype=dtype)
      self.assertEqual(dist.dtype, dtype)
      self.assertEqual(dist.dtype, dist.sample(5).dtype)

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): logits come from a float placeholder with unknown shape and the
    # distribution is sampled once.
    # Expected: feeding logits of minus 1000 and plus 1000 yields class one; feeding two batch rows with
    # opposite extremes yields classes one and zero respectively.
    with self.cached_session():
      logits = array_ops.placeholder(dtype=dtypes.float32)
      dist = categorical.Categorical(logits)
      sample = dist.sample()
      # Will sample class 1.
      sample_value = sample.eval(feed_dict={logits: [-1000.0, 1000.0]})
      self.assertEqual(1, sample_value)

      # Batch entry 0 will sample class 1, batch entry 1 will sample class 0.
      sample_value_batch = sample.eval(
          feed_dict={logits: [[-1000.0, 1000.0], [1000.0, -1000.0]]})
      self.assertAllEqual([1, 0], sample_value_batch)

  @test_util.run_deprecated_v1
  def testPMFWithBatch(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): two batch rows of class histograms (0.2, 0.8) and (0.6, 0.4) given as
    # shifted log logits.
    # Expected: the probability of events zero and one is 0.2 and 0.4 respectively.
    histograms = [[0.2, 0.8], [0.6, 0.4]]
    dist = categorical.Categorical(math_ops.log(histograms) - 50.)
    with self.cached_session():
      self.assertAllClose(dist.prob([0, 1]), [0.2, 0.4])

  @test_util.run_deprecated_v1
  def testPMFNoBatch(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a single histogram (0.2, 0.8) as shifted log logits.
    # Expected: the probability of event zero is 0.2.
    histograms = [0.2, 0.8]
    dist = categorical.Categorical(math_ops.log(histograms) - 50.)
    with self.cached_session():
      self.assertAllClose(dist.prob(0), 0.2)

  @test_util.run_deprecated_v1
  def testCDFWithDynamicEventShapeKnownNdims(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): histograms are fed through a placeholder of shape batch of two by an
    # unknown number of classes, and events through a placeholder of known length two. The same cdf op is
    # run once with three classes and once with six classes.
    # Expected: the cdf values are 0 and 1 for the three-class feed and 0.9 and 0.88 for the six-class
    # feed, showing the cdf adapts to dynamic class counts.
    """Test that dynamically-sized events with unknown shape work."""
    batch_size = 2
    histograms = array_ops.placeholder(dtype=dtypes.float32,
                                       shape=(batch_size, None))
    event = array_ops.placeholder(dtype=dtypes.float32, shape=(batch_size,))
    dist = categorical.Categorical(probs=histograms)
    cdf_op = dist.cdf(event)

    # Feed values into the placeholder with different shapes three classes.
    event_feed_one = [0, 1]
    histograms_feed_one = [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]]
    expected_cdf_one = [0.0, 1.0]
    feed_dict_one = {
        histograms: histograms_feed_one,
        event: event_feed_one
    }

    # six classes.
    event_feed_two = [2, 5]
    histograms_feed_two = [[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                           [0.15, 0.2, 0.05, 0.35, 0.13, 0.12]]
    expected_cdf_two = [0.9, 0.88]
    feed_dict_two = {
        histograms: histograms_feed_two,
        event: event_feed_two
    }

    with self.cached_session() as sess:
      actual_cdf_one = sess.run(cdf_op, feed_dict=feed_dict_one)
      actual_cdf_two = sess.run(cdf_op, feed_dict=feed_dict_two)

    self.assertAllClose(actual_cdf_one, expected_cdf_one)
    self.assertAllClose(actual_cdf_two, expected_cdf_two)

  @parameterized.named_parameters(
      ("test1", [0, 1], [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]], [0.0, 1.0]),
      ("test2", [2, 5], [[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                         [0.15, 0.2, 0.05, 0.35, 0.13, 0.12]], [0.9, 0.88]))
  def testCDFWithDynamicEventShapeUnknownNdims(
      self, events, histograms, expected_cdf):
    # [FUNCTION DOC]
    # Parameterized test with two named cases (test1 and test2) giving events, histograms and the
    # expected cdf. Inputs: events (list of class indices), histograms (nested list of class
    # probabilities) and expected_cdf (list of floats).
    # Scenario: both inputs are passed through placeholders-with-default of completely unknown shape,
    # so even their ranks are unknown.
    # Expected: evaluating the cdf returns the expected values (0 and 1 for the first case, 0.9 and
    # 0.88 for the second).
    """Test that dynamically-sized events with unknown shape work."""
    event_ph = array_ops.placeholder_with_default(events, shape=None)
    histograms_ph = array_ops.placeholder_with_default(histograms, shape=None)
    dist = categorical.Categorical(probs=histograms_ph)
    cdf_op = dist.cdf(event_ph)

    actual_cdf = self.evaluate(cdf_op)
    self.assertAllClose(actual_cdf, expected_cdf)

  @test_util.run_deprecated_v1
  def testCDFWithBatch(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): two batch rows of five-class histograms with events zero and three.
    # Expected: the cdf is 0.0 and 0.95.
    histograms = [[0.1, 0.2, 0.3, 0.25, 0.15],
                  [0.0, 0.75, 0.2, 0.05, 0.0]]
    event = [0, 3]
    expected_cdf = [0.0, 0.95]
    dist = categorical.Categorical(probs=histograms)
    cdf_op = dist.cdf(event)

    with self.cached_session():
      self.assertAllClose(cdf_op, expected_cdf)

  @test_util.run_deprecated_v1
  def testCDFNoBatch(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): one histogram over four classes and event two.
    # Expected: the cdf equals 0.3.
    histogram = [0.1, 0.2, 0.3, 0.4]
    event = 2
    expected_cdf = 0.3
    dist = categorical.Categorical(probs=histogram)
    cdf_op = dist.cdf(event)

    with self.cached_session():
      self.assertAlmostEqual(cdf_op.eval(), expected_cdf)

  @test_util.run_deprecated_v1
  def testCDFBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): histograms of shape 2x3 and an event array of shape 3x2 (an extra
    # leading batch dimension).
    # Expected: the cdf broadcasts to shape 3x2 and equals the cumulative probabilities strictly below
    # each event (0, 0, 0.2, 0.3, 0.3, 0.75).
    # shape: [batch=2, n_bins=3]
    histograms = [[0.2, 0.1, 0.7],
                  [0.3, 0.45, 0.25]]

    # shape: [batch=3, batch=2]
    devent = [
        [0, 0],
        [1, 1],
        [2, 2]
    ]
    dist = categorical.Categorical(probs=histograms)

    # We test that the probabilities are correctly broadcasted over the
    # additional leading batch dimension of size 3.
    expected_cdf_result = np.zeros((3, 2))
    expected_cdf_result[0, 0] = 0
    expected_cdf_result[0, 1] = 0
    expected_cdf_result[1, 0] = 0.2
    expected_cdf_result[1, 1] = 0.3
    expected_cdf_result[2, 0] = 0.3
    expected_cdf_result[2, 1] = 0.75

    with self.cached_session():
      self.assertAllClose(dist.cdf(devent), expected_cdf_result)

  def testBroadcastWithBatchParamsAndBiggerEvent(self):
    # [FUNCTION DOC]
    # Scenario: a Categorical with 3 batch members of four classes and integer events of shape 5x3,
    # alongside a Normal distribution with a batch of three locations and real events of shape 5x3.
    # Action: evaluates prob, log_prob, cdf and log_cdf for both distributions in one evaluation.
    # Expected: for each of the four functions the Categorical result has the same shape as the Normal
    # result, i.e. both follow the same broadcasting rules. Only shapes are compared, not values.
    ## The parameters have a single batch dimension, and the event has two.

    # param shape is [3 x 4], where 4 is the number of bins (non-batch dim).
    cat_params_py = [
        [0.2, 0.15, 0.35, 0.3],
        [0.1, 0.05, 0.68, 0.17],
        [0.1, 0.05, 0.68, 0.17]
    ]

    # event shape = [5, 3], both are "batch" dimensions.
    disc_event_py = [
        [0, 1, 2],
        [1, 2, 3],
        [0, 0, 0],
        [1, 1, 1],
        [2, 1, 0]
    ]

    # shape is [3]
    normal_params_py = [
        -10.0,
        120.0,
        50.0
    ]

    # shape is [5, 3]
    real_event_py = [
        [-1.0, 0.0, 1.0],
        [100.0, 101, -50],
        [90, 90, 90],
        [-4, -400, 20.0],
        [0.0, 0.0, 0.0]
    ]

    cat_params_tf = array_ops.constant(cat_params_py)
    disc_event_tf = array_ops.constant(disc_event_py)
    cat = categorical.Categorical(probs=cat_params_tf)

    normal_params_tf = array_ops.constant(normal_params_py)
    real_event_tf = array_ops.constant(real_event_py)
    norm = normal.Normal(loc=normal_params_tf, scale=1.0)

    # Check that normal and categorical have the same broadcasting behaviour.
    to_run = {
        "cat_prob": cat.prob(disc_event_tf),
        "cat_log_prob": cat.log_prob(disc_event_tf),
        "cat_cdf": cat.cdf(disc_event_tf),
        "cat_log_cdf": cat.log_cdf(disc_event_tf),
        "norm_prob": norm.prob(real_event_tf),
        "norm_log_prob": norm.log_prob(real_event_tf),
        "norm_cdf": norm.cdf(real_event_tf),
        "norm_log_cdf": norm.log_cdf(real_event_tf),
    }

    with self.cached_session() as sess:
      run_result = self.evaluate(to_run)

    self.assertAllEqual(run_result["cat_prob"].shape,
                        run_result["norm_prob"].shape)
    self.assertAllEqual(run_result["cat_log_prob"].shape,
                        run_result["norm_log_prob"].shape)
    self.assertAllEqual(run_result["cat_cdf"].shape,
                        run_result["norm_cdf"].shape)
    self.assertAllEqual(run_result["cat_log_cdf"].shape,
                        run_result["norm_log_cdf"].shape)

  @test_util.run_deprecated_v1
  def testLogPMF(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): two histogram rows given as shifted log logits.
    # Expected: log_prob of integer events zero and one equals the log of 0.2 and 0.4, and the same
    # holds when the events are given as floating point values 0.0 and 1.0.
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = categorical.Categorical(logits)
    with self.cached_session():
      self.assertAllClose(dist.log_prob([0, 1]), np.log([0.2, 0.4]))
      self.assertAllClose(dist.log_prob([0.0, 1.0]), np.log([0.2, 0.4]))

  @test_util.run_deprecated_v1
  def testEntropyNoBatch(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a single histogram (0.2, 0.8).
    # Expected: the entropy equals minus the sum of p log p.
    logits = np.log([0.2, 0.8]) - 50.
    dist = categorical.Categorical(logits)
    with self.cached_session():
      self.assertAllClose(dist.entropy(),
                          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)))

  @test_util.run_deprecated_v1
  def testEntropyWithBatch(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): two histogram rows (0.2, 0.8) and (0.6, 0.4).
    # Expected: the two entropies equal minus the sum of p log p for each row.
    logits = np.log([[0.2, 0.8], [0.6, 0.4]]) - 50.
    dist = categorical.Categorical(logits)
    with self.cached_session():
      self.assertAllClose(dist.entropy(), [
          -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
          -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
      ])

  @test_util.run_deprecated_v1
  def testEntropyGradient(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): fixed logits of shape 2x3; true entropy is computed with softmax and
    # log-softmax; a Categorical built from the softmax probabilities computes its own entropy.
    # Action: gradients of both entropies with respect to the logits are computed.
    # Expected: the entropy values match and the gradients match, showing the Categorical entropy is
    # differentiable through its probabilities.
    with self.cached_session() as sess:
      logits = constant_op.constant([[1., 2., 3.], [2., 5., 1.]])

      probabilities = nn_ops.softmax(logits)
      log_probabilities = nn_ops.log_softmax(logits)
      true_entropy = - math_ops.reduce_sum(
          probabilities * log_probabilities, axis=-1)

      categorical_distribution = categorical.Categorical(probs=probabilities)
      categorical_entropy = categorical_distribution.entropy()

      # works
      true_entropy_g = gradients_impl.gradients(true_entropy, [logits])
      categorical_entropy_g = gradients_impl.gradients(
          categorical_entropy, [logits])

      res = sess.run({"true_entropy": true_entropy,
                      "categorical_entropy": categorical_entropy,
                      "true_entropy_g": true_entropy_g,
                      "categorical_entropy_g": categorical_entropy_g})
      self.assertAllClose(res["true_entropy"],
                          res["categorical_entropy"])
      self.assertAllClose(res["true_entropy_g"],
                          res["categorical_entropy_g"])

  def testSample(self):
    # [FUNCTION DOC]
    # Scenario: a 1x2 batch of two-class histograms, 10000 samples with seed 123 and the static shape
    # set to 10000x1x2.
    # Expected: samples are 32-bit integers, all between 0 and 1, and the empirical frequencies of
    # classes zero and one are within 0.01 of (0.2, 0.4) and (0.8, 0.6).
    with self.cached_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)
      n = 10000
      samples = dist.sample(n, seed=123)
      samples.set_shape([n, 1, 2])
      self.assertEqual(samples.dtype, dtypes.int32)
      sample_values = self.evaluate(samples)
      self.assertFalse(np.any(sample_values < 0))
      self.assertFalse(np.any(sample_values > 1))
      self.assertAllClose(
          [[0.2, 0.4]], np.mean(
              sample_values == 0, axis=0), atol=1e-2)
      self.assertAllClose(
          [[0.8, 0.6]], np.mean(
              sample_values == 1, axis=0), atol=1e-2)

  def testSampleWithSampleShape(self):
    # [FUNCTION DOC]
    # Scenario: a 1x2 batch of two-class histograms; a sample shape of 100 by 100 is requested with
    # seed 123.
    # Action: computes the probability of each sampled value.
    # Expected: the average of the sampled probabilities for each batch member equals the sum of
    # squared class probabilities (0.2 squared plus 0.8 squared, and 0.4 squared plus 0.6 squared)
    # within 0.01.
    with self.cached_session():
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)
      samples = dist.sample((100, 100), seed=123)
      prob = dist.prob(samples)
      prob_val = self.evaluate(prob)
      self.assertAllClose(
          [0.2**2 + 0.8**2], [prob_val[:, :, :, 0].mean()], atol=1e-2)
      self.assertAllClose(
          [0.4**2 + 0.6**2], [prob_val[:, :, :, 1].mean()], atol=1e-2)

  def testNotReparameterized(self):
    # [FUNCTION DOC]
    # Scenario: probabilities are watched on a gradient tape while 100 samples are drawn.
    # Expected: the gradient of the samples with respect to the probabilities is nothing/null.
    p = constant_op.constant([0.3, 0.3, 0.4])
    with backprop.GradientTape() as tape:
      tape.watch(p)
      dist = categorical.Categorical(p)
      samples = dist.sample(100)
    grad_p = tape.gradient(samples, p)
    self.assertIsNone(grad_p)

  def testLogPMFBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario: a 1x2x2 batch (histograms with two classes) and events of ranks 0 through 3.
    # Expected: the probability result broadcasts the events against the batch, producing the correct
    # nested values for scalar, vector, matrix and higher-rank events.
    with self.cached_session():
      # 1 x 2 x 2
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)

      prob = dist.prob(1)
      self.assertAllClose([[0.8, 0.6]], self.evaluate(prob))

      prob = dist.prob([1])
      self.assertAllClose([[0.8, 0.6]], self.evaluate(prob))

      prob = dist.prob([0, 1])
      self.assertAllClose([[0.2, 0.6]], self.evaluate(prob))

      prob = dist.prob([[0, 1]])
      self.assertAllClose([[0.2, 0.6]], self.evaluate(prob))

      prob = dist.prob([[[0, 1]]])
      self.assertAllClose([[[0.2, 0.6]]], self.evaluate(prob))

      prob = dist.prob([[1, 0], [0, 1]])
      self.assertAllClose([[0.8, 0.4], [0.2, 0.6]], self.evaluate(prob))

      prob = dist.prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertAllClose([[[0.8, 0.6], [0.8, 0.4]], [[0.8, 0.4], [0.2, 0.6]]],
                          self.evaluate(prob))

  def testLogPMFShape(self):
    # [FUNCTION DOC]
    # Scenario: a 1x2x2 batch of histograms given as log logits.
    # Expected: log_prob of a vector event has rank 2 and static shape 1x2; log_prob of a 2x2x2 event
    # has rank 3 and static shape 2x2x2.
    with self.cached_session():
      # shape [1, 2, 2]
      histograms = [[[0.2, 0.8], [0.4, 0.6]]]
      dist = categorical.Categorical(math_ops.log(histograms))

      log_prob = dist.log_prob([0, 1])
      self.assertEqual(2, log_prob.get_shape().ndims)
      self.assertAllEqual([1, 2], log_prob.get_shape())

      log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
      self.assertEqual(3, log_prob.get_shape().ndims)
      self.assertAllEqual([2, 2, 2], log_prob.get_shape())

  def testLogPMFShapeNoBatch(self):
    # [FUNCTION DOC]
    # Scenario: a single two-class histogram without a batch.
    # Expected: log_prob of a scalar event is a scalar with rank zero and empty shape; log_prob of a
    # 2x2x2 event has rank 3 and static shape 2x2x2.
    histograms = [0.2, 0.8]
    dist = categorical.Categorical(math_ops.log(histograms))

    log_prob = dist.log_prob(0)
    self.assertEqual(0, log_prob.get_shape().ndims)
    self.assertAllEqual([], log_prob.get_shape())

    log_prob = dist.log_prob([[[1, 1], [1, 0]], [[1, 0], [0, 1]]])
    self.assertEqual(3, log_prob.get_shape().ndims)
    self.assertAllEqual([2, 2, 2], log_prob.get_shape())

  @test_util.run_deprecated_v1
  def testMode(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a 1x2 batch of histograms (0.2, 0.8) and (0.6, 0.4).
    # Expected: the mode is class one for the first member and class zero for the second.
    with self.cached_session():
      histograms = [[[0.2, 0.8], [0.6, 0.4]]]
      dist = categorical.Categorical(math_ops.log(histograms) - 50.)
      self.assertAllEqual(dist.mode(), [[1, 0]])

  @test_util.run_deprecated_v1
  def testCategoricalCategoricalKL(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): for 2 or 4 categories and batch sizes 1 or 10, random logits define two
    # distributions a and b.
    # Action: computes the registered KL divergence of a from b and of a from itself.
    # Expected: KL has the batch shape and equals the sum over classes of p_a times the log ratio of
    # probabilities computed with a numpy softmax; KL of a distribution with itself is zero.

    def np_softmax(logits):
      # [FUNCTION DOC]
      # HELPER: numpy softmax used by testCategoricalCategoricalKL to get reference probabilities.
      # Takes an array of logits and returns exponentials normalized to sum to one over the last axis.
      # Not numerically stabilized by subtracting the maximum, so large logits could overflow. Pure.
      exp_logits = np.exp(logits)
      return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    with self.cached_session() as sess:
      for categories in [2, 4]:
        for batch_size in [1, 10]:
          a_logits = np.random.randn(batch_size, categories)
          b_logits = np.random.randn(batch_size, categories)

          a = categorical.Categorical(logits=a_logits)
          b = categorical.Categorical(logits=b_logits)

          kl = kullback_leibler.kl_divergence(a, b)
          kl_val = self.evaluate(kl)
          # Make sure KL(a||a) is 0
          kl_same = sess.run(kullback_leibler.kl_divergence(a, a))

          prob_a = np_softmax(a_logits)
          prob_b = np_softmax(b_logits)
          kl_expected = np.sum(prob_a * (np.log(prob_a) - np.log(prob_b)),
                               axis=-1)

          self.assertEqual(kl.get_shape(), (batch_size,))
          self.assertAllClose(kl_val, kl_expected)
          self.assertAllClose(kl_same, np.zeros_like(kl_expected))


if __name__ == "__main__":
  test.main()
