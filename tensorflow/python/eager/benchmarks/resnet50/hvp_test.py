# [MODULE DOC]
# Tests and benchmarks for Hessian-vector products computed on a ResNet50 model with four
# strategies: forward-over-backward using a forward accumulator, back-over-forward, forward
# over backward built from graph-style gradients, and back-over-back using nested tapes. Each
# strategy is a module-level helper taking a model, image batch, label batch and tangent vector
# list. The test class checks that every strategy, eagerly and compiled, yields results shaped
# and typed like the model's trainable variables; the benchmark class times them. Uses the
# shared resnet50 model and test utilities. Running as a script enables v2 behavior and runs.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for Hessian-vector products with ResNet50."""

import gc
import time

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.eager import forwardprop
from tensorflow.python.eager.benchmarks.resnet50 import resnet50
from tensorflow.python.eager.benchmarks.resnet50 import resnet50_test_util


def _forward_over_back_hvp(model, images, labels, vector):
  # [FUNCTION DOC]
  # HELPER: one Hessian-vector-product strategy shared by the tests and benchmarks.
  # Computes the Hessian-vector product of the softmax cross-entropy loss of the model on the
  # given images and one-hot labels, by recording a forward accumulator (tangents given by the
  # list vector, one per trainable variable) around a gradient tape, then taking the
  # forward-mode derivative of the gradients. Returns a list with one tensor per trainable
  # variable. The model is run in training mode.
  with forwardprop.ForwardAccumulator(
      model.trainable_variables, vector) as acc:
    with tf.GradientTape() as grad_tape:
      logits = model(images, training=True)
      loss = tf.compat.v1.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels)
    grads = grad_tape.gradient(loss, model.trainable_variables)
  return acc.jvp(grads)


def _back_over_forward_hvp(model, images, labels, vector):
  # [FUNCTION DOC]
  # HELPER: one Hessian-vector-product strategy shared by the tests and benchmarks.
  # Watches the trainable variables on a gradient tape, runs the model in training mode under a
  # forward accumulator with tangents from the list vector, computes the softmax cross-entropy
  # loss, and returns the reverse-mode gradient of the loss's forward-mode derivative with
  # respect to the trainable variables: a list with one tensor per variable.
  with tf.GradientTape() as grad_tape:
    grad_tape.watch(model.trainable_variables)
    with forwardprop.ForwardAccumulator(
        model.trainable_variables, vector) as acc:
      logits = model(images, training=True)
      loss = tf.compat.v1.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels)
  return grad_tape.gradient(acc.jvp(loss), model.trainable_variables)


def _tf_gradients_forward_over_back_hvp(model, images, labels, vector):
  # [FUNCTION DOC]
  # HELPER: one Hessian-vector-product strategy shared by the tests and benchmarks; meant to
  # be wrapped in a compiled function because it uses graph-mode gradient calls.
  # Computes gradients of the softmax cross-entropy loss with a tape, then emulates forward
  # mode with the double-backward trick: gradients of the gradients with respect to the
  # variables using dummy all-ones helper tensors, then gradients of that with respect to the
  # helpers using the tangent list vector. Returns a list of tensors, one per trainable
  # variable.
  with tf.GradientTape() as grad_tape:
    logits = model(images, training=True)
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)
  variables = model.trainable_variables
  grads = grad_tape.gradient(loss, variables)
  helpers = tf.nest.map_structure(tf.ones_like, grads)
  transposing = tf.gradients(grads, variables, helpers)
  return tf.gradients(transposing, helpers, vector)


def _back_over_back_hvp(model, images, labels, vector):
  # [FUNCTION DOC]
  # HELPER: one Hessian-vector-product strategy shared by the tests and benchmarks.
  # Computes the softmax cross-entropy loss under an inner gradient tape nested in an outer
  # tape, takes the inner gradient with respect to the trainable variables, then the outer
  # gradient of those gradients using the list vector as output gradients. Returns a list with
  # one tensor per trainable variable.
  with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
      logits = model(images, training=True)
      loss = tf.compat.v1.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels)
    grads = inner_tape.gradient(loss, model.trainable_variables)
  return outer_tape.gradient(
      grads, model.trainable_variables, output_gradients=vector)


class HVPTest(tf.test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case checking that each Hessian-vector-product strategy, eager and
  # compiled, produces output matching the model's trainable variables in shape and dtype.

  @parameterized.named_parameters(
      ("forward_over_back_eager", _forward_over_back_hvp),
      ("forward_over_back_function", tf.function(_forward_over_back_hvp)),
      ("tf_gradients", tf.function(_tf_gradients_forward_over_back_hvp)),
      ("back_over_back_eager", _back_over_back_hvp),
      ("back_over_back_function", tf.function(_back_over_back_hvp)),
      ("back_over_forward_eager", _back_over_forward_hvp),
      ("back_over_forward_function", tf.function(_back_over_forward_hvp)))
  def test_hvp_shapes(self, hvp_function):
    # [FUNCTION DOC]
    # Parameterized over seven eager or compiled strategy functions (hvp_function). Builds a
    # ResNet50 on the preferred device and data layout, creates a random batch of two images
    # and labels, builds the model, uses an all-ones tangent per trainable variable, calls the
    # strategy, and asserts each result component has the same shape and dtype as the
    # corresponding variable. Numerical values are not checked.
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format)
    with tf.device(device):
      images, labels = resnet50_test_util.random_batch(2, data_format)
      images = tf.constant(images)
      labels = tf.constant(labels)
      model.build(images.shape)
      vector = [tf.ones_like(v) for v in model.trainable_variables]

      # Note that numerical differences build up to quite large differences here
      # in the final hvp. tensorflow/python/eager:forwardprop_test has a
      # smaller-scale test that the computations are close on a much smaller but
      # otherwise similar model.
      hvp = hvp_function(model, images, labels, vector)
      for hvp_component, variable in zip(hvp, model.trainable_variables):
        self.assertEqual(hvp_component.shape, variable.shape)
        self.assertEqual(hvp_component.dtype, variable.dtype)


class HVPBenchmarks(tf.test.Benchmark):
  # [CLASS DOC]
  # Benchmarks timing each Hessian-vector-product strategy (eager and compiled) on ResNet50
  # with a batch of eight, using a shared timing helper.

  def _force_device_sync(self):
    # [FUNCTION DOC]
    # HELPER: synchronization utility for the timing helper.
    # Creates a scalar constant and copies it to the host; when called under a non-CPU device
    # scope this forces pending device work to finish. Returns nothing.
    # If this function is called in the context of a non-CPU device
    # (e.g., inside a 'with tf.device("/gpu:0")' block)
    # then this will force a copy from CPU->NON_CPU_DEVICE->CPU,
    # which forces a sync. This is a roundabout way, yes.
    tf.constant(1.).cpu()

  def _hvp_benchmark(self, hvp_fn, label, batch_sizes,
                     num_iters=30, num_burn=5):
    # [FUNCTION DOC]
    # HELPER: shared benchmark driver for all strategies.
    # Inputs: hvp_fn, a strategy callable taking model, images, labels and tangent list; label,
    # a name for the report; batch_sizes, list of batch sizes; num_iters (default 30) timed
    # iterations; num_burn (default 5) untimed warm-up iterations. Builds one ResNet50, and for
    # each batch size creates random data on the preferred device, runs warm-up iterations,
    # syncs, collects garbage, times the iterations (copying every result to the host), syncs
    # again and reports through the shared report helper. Returns nothing.
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format)
    for batch_size in batch_sizes:
      with tf.device(device):
        images, labels = resnet50_test_util.random_batch(
            batch_size, data_format)
        images = tf.constant(images)
        labels = tf.constant(labels)
        model.build(images.shape)
        vector = [tf.ones_like(v) for v in model.trainable_variables]
        for _ in range(num_burn):
          results = hvp_fn(model, images, labels, vector)
          for result in results:
            result.cpu()
        self._force_device_sync()
        gc.collect()
        start = time.time()
        for _ in range(num_iters):
          results = hvp_fn(model, images, labels, vector)
          for result in results:
            result.cpu()
        self._force_device_sync()
        resnet50_test_util.report(
            self, label, start, num_iters, device, batch_size, data_format)

  def benchmark_forward_over_backward_hvp_eager(self):
    # [FUNCTION DOC]
    # Benchmark of the eager forward-over-backward strategy with batch size eight.
    self._hvp_benchmark(_forward_over_back_hvp,
                        "forward_over_backward_hvp_eager",
                        batch_sizes=[8])

  def benchmark_forward_over_backward_hvp_function(self):
    # [FUNCTION DOC]
    # Benchmark of the compiled forward-over-backward strategy with batch size eight.
    self._hvp_benchmark(tf.function(_forward_over_back_hvp),
                        "forward_over_backward_hvp_function",
                        batch_sizes=[8])

  def benchmark_tf_gradients_forward_over_backward_hvp_function(self):
    # [FUNCTION DOC]
    # Benchmark of the compiled graph-gradients forward-over-backward strategy, batch size
    # eight.
    self._hvp_benchmark(tf.function(_tf_gradients_forward_over_back_hvp),
                        "tf_gradients_forward_over_backward_hvp_function",
                        batch_sizes=[8])

  def benchmark_backward_over_backward_hvp_eager(self):
    # [FUNCTION DOC]
    # Benchmark of the eager back-over-back (nested tapes) strategy with batch size eight.
    self._hvp_benchmark(_back_over_back_hvp,
                        "backward_over_backward_hvp_eager",
                        batch_sizes=[8])

  def benchmark_backward_over_backward_hvp_function(self):
    # [FUNCTION DOC]
    # Benchmark of the compiled back-over-back (nested tapes) strategy with batch size eight.
    self._hvp_benchmark(tf.function(_back_over_back_hvp),
                        "backward_over_backward_hvp_function",
                        batch_sizes=[8])

  def benchmark_backward_over_forward_hvp_eager(self):
    # [FUNCTION DOC]
    # Benchmark of the eager back-over-forward strategy with batch size eight.
    self._hvp_benchmark(_back_over_forward_hvp,
                        "backward_over_forward_hvp_eager",
                        batch_sizes=[8])

  def benchmark_backward_over_forward_hvp_function(self):
    # [FUNCTION DOC]
    # Benchmark of the compiled back-over-forward strategy with batch size eight.
    self._hvp_benchmark(tf.function(_back_over_forward_hvp),
                        "backward_over_forward_hvp_function",
                        batch_sizes=[8])


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
