# [MODULE DOC]
# Test and benchmarks of the ResNet50 model under graph (session-based) execution using the
# v1 compatibility API. Provides small helpers choosing the data layout (channels-first when a
# GPU is available), the image shape and a random image/one-hot-label batch, one test checking
# inference output shape, and a benchmark class timing inference and training steps. Uses the
# model from the sibling resnet50 module. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for ResNet50 under graph execution."""

import time

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.eager.benchmarks.resnet50 import resnet50


def data_format():
  # [FUNCTION DOC]
  # HELPER: layout chooser for the tests and benchmarks in this module.
  # Returns the string for channels-first layout if a GPU is available, otherwise the string for
  # channels-last layout. No arguments, no side effects beyond a device query.
  return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'


def image_shape(batch_size):
  # [FUNCTION DOC]
  # HELPER: shape computation for the tests and benchmarks in this module.
  # Input: batch_size, an integer or nothing (for a placeholder with unknown batch dimension).
  # Returns a list of four items: the batch size and 3 by 224 by 224 for channels-first, or
  # 224 by 224 by 3 after the batch size for channels-last, according to the chosen layout.
  if data_format() == 'channels_first':
    return [batch_size, 3, 224, 224]
  return [batch_size, 224, 224, 3]


def random_batch(batch_size):
  # [FUNCTION DOC]
  # HELPER: synthetic data generator for the tests and benchmarks in this module.
  # Input: batch_size, integer number of examples. Returns a pair of numpy arrays: uniformly
  # random float32 images in the layout chosen by the data format helper, and float32 one-hot
  # labels of shape batch size by 1000 built from random integer class ids. Uses the global
  # numpy random state (no seed control).
  images = np.random.rand(*image_shape(batch_size)).astype(np.float32)
  num_classes = 1000
  labels = np.random.randint(
      low=0, high=num_classes, size=[batch_size]).astype(np.int32)
  one_hot = np.zeros((batch_size, num_classes)).astype(np.float32)
  one_hot[np.arange(batch_size), labels] = 1.
  return images, one_hot


class ResNet50GraphTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case for running ResNet50 inference in a v1 graph and session.

  def testApply(self):
    # [FUNCTION DOC]
    # Test: in a fresh graph, builds an image placeholder with unknown batch size and a ResNet50
    # called in inference mode, initializes all variables in a session, feeds a random batch of
    # eight images and asserts the output shape is eight by 1000 logits.
    # Use small batches for tests because the OSS version runs
    # in constrained GPU environment with 1-2GB of memory.
    batch_size = 8
    with tf.Graph().as_default():
      images = tf.placeholder(tf.float32, image_shape(None))
      model = resnet50.ResNet50(data_format())
      predictions = model(images, training=False)

      init = tf.global_variables_initializer()

      with tf.Session() as sess:
        sess.run(init)
        np_images, _ = random_batch(batch_size)
        out = sess.run(predictions, feed_dict={images: np_images})
        self.assertAllEqual([batch_size, 1000], out.shape)


class ResNet50Benchmarks(tf.test.Benchmark):
  # [CLASS DOC]
  # Benchmarks of ResNet50 inference and training under graph execution, reporting average
  # step time and examples per second.

  def _report(self, label, start, num_iters, batch_size):
    # [FUNCTION DOC]
    # HELPER: result reporter for the benchmarks of this class.
    # Inputs: label, short name of the workload; start, wall-clock start time of the timed loop;
    # num_iters, number of timed iterations; batch_size, examples per iteration. Computes the
    # average time per iteration up to now, builds a benchmark name from graph, label, device
    # kind (gpu or cpu), batch size and layout, and records it with the iteration count, wall
    # time and an examples-per-second extra. Returns nothing.
    avg_time = (time.time() - start) / num_iters
    dev = 'gpu' if tf.test.is_gpu_available() else 'cpu'
    name = 'graph_%s_%s_batch_%d_%s' % (label, dev, batch_size, data_format())
    extras = {'examples_per_sec': batch_size / avg_time}
    self.report_benchmark(
        iters=num_iters, wall_time=avg_time, name=name, extras=extras)

  def benchmark_graph_apply(self):
    # [FUNCTION DOC]
    # Benchmark of graph-mode inference with batch size 64: builds the model on a placeholder,
    # initializes variables, runs three untimed warm-up steps, times thirty steps feeding the same
    # random batch (so it includes host-to-device copy of the feed), and reports the result.
    with tf.Graph().as_default():
      images = tf.placeholder(tf.float32, image_shape(None))
      model = resnet50.ResNet50(data_format())
      predictions = model(images, training=False)

      init = tf.global_variables_initializer()

      batch_size = 64
      with tf.Session() as sess:
        sess.run(init)
        np_images, _ = random_batch(batch_size)
        num_burn, num_iters = (3, 30)
        for _ in range(num_burn):
          sess.run(predictions, feed_dict={images: np_images})
        start = time.time()
        for _ in range(num_iters):
          # Comparison with the eager execution benchmark in resnet50_test.py
          # isn't entirely fair as the time here includes the cost of copying
          # the feeds from CPU memory to GPU.
          sess.run(predictions, feed_dict={images: np_images})
        self._report('apply', start, num_iters, batch_size)

  def benchmark_graph_train(self):
    # [FUNCTION DOC]
    # Benchmark of graph-mode training for batch sizes 16, 32 and 64. For each, builds a fresh
    # graph with a repeating one-shot dataset iterator over one random batch, a ResNet50 in
    # training mode, softmax cross-entropy loss and a plain gradient-descent step with learning
    # rate one, initializes variables, runs five warm-up and ten timed training steps and reports.
    for batch_size in [16, 32, 64]:
      with tf.Graph().as_default():
        np_images, np_labels = random_batch(batch_size)
        dataset = tf.data.Dataset.from_tensors((np_images, np_labels)).repeat()
        images, labels = tf.data.make_one_shot_iterator(
            dataset).get_next()

        model = resnet50.ResNet50(data_format())
        logits = model(images, training=True)
        loss = tf.compat.v1.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
          sess.run(init)
          (num_burn, num_iters) = (5, 10)
          for _ in range(num_burn):
            sess.run(train_op)
          start = time.time()
          for _ in range(num_iters):
            sess.run(train_op)
          self._report('train', start, num_iters, batch_size)


if __name__ == '__main__':
  tf.test.main()
