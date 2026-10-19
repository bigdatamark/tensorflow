# [MODULE DOC]
# Tests and benchmarks for the ResNet50 model under eager execution. The tests check output
# shapes for several model configurations (with or without the classifier, pooling, stride
# changes, intermediate feature capture), that one training step works eagerly (also async and
# compiled) and writes a loss summary event, and that repeated training steps leave no garbage
# for the collector. The benchmarks time inference and training, eager, async, compiled and
# with datasets. Also exposes gradient helpers, summary event-reading helpers and a small mock
# iterator. Uses the sibling resnet50 model and test utility modules. Running as a script
# enables eager execution and runs the tests.
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
"""Tests and benchmarks for the ResNet50 model, executed eagerly."""

import gc
import os
import tempfile
import time

import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.benchmarks.resnet50 import resnet50
from tensorflow.python.eager.benchmarks.resnet50 import resnet50_test_util
from tensorflow.python.framework import test_util


def compute_gradients(model, images, labels, num_replicas=1):
  # [FUNCTION DOC]
  # HELPER: training-step helper shared by the tests and benchmarks.
  # Inputs: model, a ResNet50 model; images, batch of images; labels, one-hot labels;
  # num_replicas, replica count (default one) by which the loss is divided when not one. Runs
  # the model in training mode on a gradient tape, computes softmax cross-entropy loss, writes
  # the unscaled loss under the name loss to the active summary writer (the write happens before
  # scaling), then computes gradients of the (scaled) loss with respect to all model variables
  # with recording stopped as a workaround for a multi-threading tracing bug. Returns the list
  # of gradients.
  with tf.GradientTape() as grad_tape:
    logits = model(images, training=True)
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)
    tf.compat.v2.summary.write('loss', loss)
    if num_replicas != 1:
      loss /= num_replicas

  # TODO(b/110991947): We can mistakenly trace the gradient call in
  # multi-threaded environment. Explicitly disable recording until
  # this is fixed.
  with record.stop_recording():
    grads = grad_tape.gradient(loss, model.variables)
  return grads


def apply_gradients(model, optimizer, gradients):
  # [FUNCTION DOC]
  # HELPER: training-step helper shared by the tests and benchmarks.
  # Inputs: model, whose variables are updated; optimizer, an object with an apply-gradients
  # method; gradients, list aligned with the model's variables. Pairs each gradient with its
  # variable and applies the update (side effect: variables change). Returns nothing.
  optimizer.apply_gradients(zip(gradients, model.variables))


def _events_from_file(filepath):
  # [FUNCTION DOC]
  # HELPER: reader used by the events-from-logdir helper.
  # Input: filepath, path of one summary event file. Reads every record with the v1 record
  # iterator, parses each into an event message and returns the list of events in file order.
  # Raises whatever the file reading or parsing raises (missing file, corrupt record).
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.compat.v1.Event protos in the event file.
  """
  records = list(tf.compat.v1.python_io.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = tf.compat.v1.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def events_from_logdir(logdir):
  # [FUNCTION DOC]
  # HELPER: lets tests inspect summaries written to a log directory.
  # Input: logdir, directory path expected to exist and contain exactly one file. Returns the
  # list of event messages parsed from that file. Raises an assertion error if the directory
  # does not exist or does not hold exactly one file (note: these are plain assertions, so they
  # vanish when assertions are disabled).
  """Returns all events in the single eventfile in logdir.

  Args:
    logdir: The directory in which the single event file is sought.

  Returns:
    A list of all tf.compat.v1.Event protos from the single event file.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert tf.io.gfile.exists(logdir)
  files = tf.io.gfile.listdir(logdir)
  assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
  return _events_from_file(os.path.join(logdir, files[0]))


class ResNet50Test(tf.test.TestCase):
  # [CLASS DOC]
  # Test case for the eager ResNet50 model: inference output shapes for several configurations,
  # one training step with summary output, and absence of reference-cycle garbage.

  def _apply(self, defun=False, execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the apply tests.
    # Inputs: defun, whether to replace the model's call with a compiled function (default
    # false); execution_mode, optional eager execution mode (sync or async). Builds a default
    # ResNet50 on the preferred device, runs inference on a random batch of two images, waits
    # for async work, and asserts the output shape is two by 1000.
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format)
    if defun:
      model.call = tf.function(model.call)
    with tf.device(device), context.execution_mode(execution_mode):
      images, _ = resnet50_test_util.random_batch(2, data_format)
      output = model(images, training=False)
      context.async_wait()
    self.assertEqual((2, 1000), output.shape)

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_apply(self):
    # [FUNCTION DOC]
    # Test: eager synchronous inference without compilation (disabled under the TFRT runtime
    # as flaky); expects output shape two by 1000.
    self._apply(defun=False)

  def test_apply_async(self):
    # [FUNCTION DOC]
    # Test: eager inference with asynchronous execution mode and no compilation; expects
    # output shape two by 1000.
    self._apply(defun=False, execution_mode=context.ASYNC)

  def test_apply_with_defun(self):
    # [FUNCTION DOC]
    # Test: inference with the model's call wrapped in a compiled function, synchronous mode;
    # expects output shape two by 1000.
    self._apply(defun=True)

  def test_apply_with_defun_async(self):
    # [FUNCTION DOC]
    # Test: inference with a compiled call in asynchronous execution mode; expects output
    # shape two by 1000.
    self._apply(defun=True, execution_mode=context.ASYNC)

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_apply_no_top(self):
    # [FUNCTION DOC]
    # Test: model without the top classifier and without pooling; asserts the output is the
    # last feature map of shape two, 2048 channels, one by one (layout dependent).
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format, include_top=False)
    with tf.device(device):
      images, _ = resnet50_test_util.random_batch(2, data_format)
      output = model(images, training=False)
    output_shape = ((2, 2048, 1, 1)
                    if data_format == 'channels_first' else (2, 1, 1, 2048))
    self.assertEqual(output_shape, output.shape)

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_apply_with_pooling(self):
    # [FUNCTION DOC]
    # Test: model without the classifier but with average global pooling; asserts the output
    # shape is two by 2048.
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format, include_top=False, pooling='avg')
    with tf.device(device):
      images, _ = resnet50_test_util.random_batch(2, data_format)
      output = model(images, training=False)
    self.assertEqual((2, 2048), output.shape)

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_apply_no_average_pooling(self):
    # [FUNCTION DOC]
    # Test: model without classifier and with the stage-five average pooling disabled; asserts
    # the output feature map is two, 2048 channels, seven by seven (layout dependent).
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(
        data_format, average_pooling=False, include_top=False)
    with tf.device(device):
      images, _ = resnet50_test_util.random_batch(2, data_format)
      output = model(images, training=False)
    output_shape = ((2, 2048, 7, 7) if data_format == 'channels_first' else
                    (2, 7, 7, 2048))
    self.assertEqual(output_shape, output.shape)

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_apply_block3_strides(self):
    # [FUNCTION DOC]
    # Test: model without classifier and with the extra stride after stage four enabled;
    # asserts the output is two, 2048 channels, one by one (layout dependent).
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(
        data_format, block3_strides=True, include_top=False)
    with tf.device(device):
      images, _ = resnet50_test_util.random_batch(2, data_format)
      output = model(images, training=False)
    output_shape = ((2, 2048, 1, 1) if data_format == 'channels_first' else
                    (2, 1, 1, 2048))
    self.assertEqual(output_shape, output.shape)

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_apply_retrieve_intermediates(self):
    # [FUNCTION DOC]
    # Test: model with the extra stride and no classifier, called with a mapping to collect
    # intermediate feature maps. Asserts the final output shape and that every recorded block
    # (block0, block0mp, block1 to block4) has the expected shape for the active data layout,
    # for example 64 channels at 112 by 112 for the first block and 1024 at 7 by 7 for block3.
    # Only recorded entries are checked; missing entries would not fail.
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(
        data_format, block3_strides=True, include_top=False)
    intermediates_dict = {}
    with tf.device(device):
      images, _ = resnet50_test_util.random_batch(2, data_format)
      output = model(images, training=False,
                     intermediates_dict=intermediates_dict)
    output_shape = ((2, 2048, 1, 1) if data_format == 'channels_first' else
                    (2, 1, 1, 2048))
    self.assertEqual(output_shape, output.shape)

    if data_format == 'channels_first':
      block_shapes = {
          'block0': (2, 64, 112, 112),
          'block0mp': (2, 64, 55, 55),
          'block1': (2, 256, 55, 55),
          'block2': (2, 512, 28, 28),
          'block3': (2, 1024, 7, 7),
          'block4': (2, 2048, 1, 1),
      }
    else:
      block_shapes = {
          'block0': (2, 112, 112, 64),
          'block0mp': (2, 55, 55, 64),
          'block1': (2, 55, 55, 256),
          'block2': (2, 28, 28, 512),
          'block3': (2, 7, 7, 1024),
          'block4': (2, 1, 1, 2048),
      }
    for (block_name, block) in intermediates_dict.items():
      self.assertEqual(block_shapes[block_name], block.shape)

  def _test_train(self, execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the training tests.
    # Input: execution_mode, optional eager execution mode. Sets the summary step to the global
    # step, creates a summary writer in a fresh temporary directory (unbuffered), and under the
    # preferred device and mode performs one gradient-descent training step (learning rate 0.1)
    # on a random batch of two. Asserts the model has 320 variables, then that the log directory
    # holds two events (file header and one summary) whose second carries the tag loss.
    # Side effects: creates a temporary directory (not removed), writes a summary file.
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format)
    tf.compat.v2.summary.experimental.set_step(
        tf.compat.v1.train.get_or_create_global_step())
    logdir = tempfile.mkdtemp()
    with tf.compat.v2.summary.create_file_writer(
        logdir, max_queue=0,
        name='t0').as_default(), tf.compat.v2.summary.record_if(True):
      with tf.device(device), context.execution_mode(execution_mode):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
        images, labels = resnet50_test_util.random_batch(2, data_format)
        apply_gradients(model, optimizer,
                        compute_gradients(model, images, labels))
        self.assertEqual(320, len(model.variables))
        context.async_wait()
    events = events_from_logdir(logdir)
    self.assertEqual(len(events), 2)
    self.assertEqual(events[1].summary.value[0].tag, 'loss')

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_train(self):
    # [FUNCTION DOC]
    # Test: one synchronous eager training step writing a loss summary (see the shared body).
    self._test_train()

  @test_util.disable_tfrt('TFE_ContextGetExecutorForThread missing b/156188669')
  def test_train_async(self):
    # [FUNCTION DOC]
    # Test: one training step in asynchronous execution mode writing a loss summary.
    self._test_train(execution_mode=context.ASYNC)

  @test_util.disable_tfrt('Flaky test. b/157103729')
  def test_no_garbage(self):
    # [FUNCTION DOC]
    # Test that repeated training steps create no uncollectable garbage. Disables automatic
    # garbage collection, does one warm-up step and a collection, turns on save-all debugging,
    # runs two more training steps, collects and asserts the saved-garbage list is empty. Then
    # restores the previous debug flags and re-enables collection (not restored if an assertion
    # fails earlier).
    device, data_format = resnet50_test_util.device_and_data_format()
    model = resnet50.ResNet50(data_format)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    with tf.device(device):
      images, labels = resnet50_test_util.random_batch(2, data_format)
      gc.disable()
      # Warm up. Note that this first run does create significant amounts of
      # garbage to be collected. The hope is that this is a build-only effect,
      # and a subsequent training loop will create nothing which needs to be
      # collected.
      apply_gradients(model, optimizer,
                      compute_gradients(model, images, labels))
      gc.collect()
      previous_gc_debug_flags = gc.get_debug()
      gc.set_debug(gc.DEBUG_SAVEALL)
      for _ in range(2):
        # Run twice to ensure that garbage that is created on the first
        # iteration is no longer accessible.
        apply_gradients(model, optimizer,
                        compute_gradients(model, images, labels))
      gc.collect()
      # There should be no garbage requiring collection.
      self.assertEqual(0, len(gc.garbage))
      gc.set_debug(previous_gc_debug_flags)
      gc.enable()


class MockIterator(object):
  # [CLASS DOC]
  # HELPER: minimal stand-in for a dataset iterator used by the training benchmarks. Holds a
  # fixed list of tensors and hands back that same list on every request.

  def __init__(self, tensors):
    # [FUNCTION DOC]
    # Constructor. Input: tensors, an iterable of tensors (such as images and labels). Stores a
    # list of identity copies of them (made once, on the current device).
    self._tensors = [tf.identity(x) for x in tensors]

  def next(self):
    # [FUNCTION DOC]
    # Returns the stored list of tensors; the same list object every call and never exhausted.
    return self._tensors


class ResNet50Benchmarks(tf.test.Benchmark):
  # [CLASS DOC]
  # Benchmarks of eager ResNet50 inference and training (sync, async, compiled, and with mock or
  # real dataset iterators), reporting through the shared report helper.

  def _report(self, label, start, num_iters, device, batch_size, data_format,
              num_replicas=1):
    # [FUNCTION DOC]
    # HELPER: forwards its arguments unchanged to the shared report helper, with this benchmark
    # object as the first argument, to record timing and examples per second. Returns nothing.
    resnet50_test_util.report(self, label, start, num_iters, device, batch_size,
                              data_format, num_replicas)

  def _train_batch_sizes(self):
    # [FUNCTION DOC]
    # HELPER: picks batch sizes for the training benchmarks. Scans local devices: on a GPU
    # whose description mentions K20 or P1000 returns sixteen only, on a P100 returns 16, 32 and
    # 64; on a TPU returns 32; otherwise after the scan returns 16 and 32. Returns a tuple of
    # integers. GPUs of other kinds fall through to later devices or the default. Selection is by
    # substring match on the device description, not by measured capability.
    """Choose batch sizes based on GPU capability."""
    for device in device_lib.list_local_devices():
      # TODO(b/141475121): We need some way to check which batch sizes would
      # work using a public API.
      if tf.DeviceSpec.from_string(device.name).device_type == 'GPU':
        # Avoid OOM errors with larger batch sizes, which seem to cause errors
        # later on even if caught.
        #
        # TODO(allenl): Base this on device memory; memory limit information
        # during the test seems to exclude the amount TensorFlow has allocated,
        # which isn't useful.
        if 'K20' in device.physical_device_desc:
          return (16,)
        # Quardro P1000.
        if 'P1000' in device.physical_device_desc:
          return (16,)
        if 'P100' in device.physical_device_desc:
          return (16, 32, 64)

      if tf.DeviceSpec.from_string(device.name).device_type == 'TPU':
        return (32,)
    return (16, 32)

  def _force_device_sync(self):
    # [FUNCTION DOC]
    # HELPER: synchronization utility for the timing code. Creates a scalar constant and copies
    # it to the host, which under a non-CPU device scope forces pending device work to finish.
    # Returns nothing.
    # If this function is called in the context of a non-CPU device
    # (e.g., inside a 'with tf.device("/gpu:0")' block)
    # then this will force a copy from CPU->NON_CPU_DEVICE->CPU,
    # which forces a sync. This is a roundabout way, yes.
    tf.constant(1.).cpu()

  def _benchmark_eager_apply(self, label, device_and_format, defun=False,
                             execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the inference benchmarks.
    # Inputs: label, report name; device_and_format, a pair of device name and data layout;
    # defun, whether to compile the model call (default false); execution_mode, optional eager
    # mode. Builds a ResNet50, uses batch size 64, runs five untimed warm-up and thirty timed
    # inference iterations (each result copied to the host), waiting for async work when a mode
    # is set, collecting garbage before timing, and reports the result. Returns nothing.
    with context.execution_mode(execution_mode):
      device, data_format = device_and_format
      model = resnet50.ResNet50(data_format)
      if defun:
        model.call = tf.function(model.call)
      batch_size = 64
      num_burn = 5
      num_iters = 30
      with tf.device(device):
        images, _ = resnet50_test_util.random_batch(batch_size, data_format)
        for _ in range(num_burn):
          model(images, training=False).cpu()
        if execution_mode:
          context.async_wait()
        gc.collect()
        start = time.time()
        for _ in range(num_iters):
          model(images, training=False).cpu()
        if execution_mode:
          context.async_wait()
        self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_apply_sync(self):
    # [FUNCTION DOC]
    # Benchmark: eager synchronous inference without compilation.
    self._benchmark_eager_apply(
        'eager_apply', resnet50_test_util.device_and_data_format(),
        defun=False)

  def benchmark_eager_apply_async(self):
    # [FUNCTION DOC]
    # Benchmark: eager inference in asynchronous execution mode without compilation.
    self._benchmark_eager_apply(
        'eager_apply_async',
        resnet50_test_util.device_and_data_format(),
        defun=False,
        execution_mode=context.ASYNC)

  def benchmark_eager_apply_with_defun(self):
    # [FUNCTION DOC]
    # Benchmark: inference with the model call compiled, synchronous mode.
    self._benchmark_eager_apply(
        'eager_apply_with_defun',
        resnet50_test_util.device_and_data_format(), defun=True)

  def _benchmark_eager_train(self,
                             label,
                             make_iterator,
                             device_and_format,
                             defun=False,
                             execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the training benchmarks.
    # Inputs: label, report name; make_iterator, a callable taking an images-and-labels pair and
    # returning an object with a next method (or equivalent) that yields batches; device_and_format,
    # pair of device name and layout; defun, whether to compile the model call and the apply
    # step (default false); execution_mode, optional eager mode. For each batch size from the
    # batch-size helper, builds a random batch, a fresh model and a momentum optimizer (rate 0.1,
    # momentum 0.1; momentum is set to satisfy a lowering pipeline), runs three warm-up and ten
    # timed training steps pulling batches from the iterator, syncing devices and waiting for
    # async work, and reports. Returns nothing.
    with context.execution_mode(execution_mode):
      device, data_format = device_and_format
      for batch_size in self._train_batch_sizes():
        (images, labels) = resnet50_test_util.random_batch(
            batch_size, data_format)
        model = resnet50.ResNet50(data_format)
        # TODO(b/161911585): tf_to_corert MLIR lowering pipeline should handle
        # case when momentum is not set.
        optimizer = tf.keras.optimizers.SGD(0.1, 0.1)
        apply_grads = apply_gradients
        if defun:
          model.call = tf.function(model.call)
          apply_grads = tf.function(apply_gradients)

        num_burn = 3
        num_iters = 10
        with tf.device(device):
          iterator = make_iterator((images, labels))
          for _ in range(num_burn):
            (images, labels) = iterator.next()
            apply_grads(model, optimizer,
                        compute_gradients(model, images, labels))
          if execution_mode:
            context.async_wait()
          self._force_device_sync()
          gc.collect()

          start = time.time()
          for _ in range(num_iters):
            (images, labels) = iterator.next()
            apply_grads(model, optimizer,
                        compute_gradients(model, images, labels))
          if execution_mode:
            context.async_wait()
          self._force_device_sync()
          self._report(label, start, num_iters, device, batch_size, data_format)

  def benchmark_eager_train_sync(self):
    # [FUNCTION DOC]
    # Benchmark: eager synchronous training using the mock iterator, no compilation.
    self._benchmark_eager_train(
        'eager_train', MockIterator,
        resnet50_test_util.device_and_data_format(), defun=False)

  def benchmark_eager_train_async(self):
    # [FUNCTION DOC]
    # Benchmark: eager training in asynchronous execution mode using the mock iterator.
    self._benchmark_eager_train(
        'eager_train_async',
        MockIterator,
        resnet50_test_util.device_and_data_format(),
        defun=False,
        execution_mode=context.ASYNC)

  def benchmark_eager_train_with_defun(self):
    # [FUNCTION DOC]
    # Benchmark: training with compiled model call and apply step, using the mock iterator.
    self._benchmark_eager_train(
        'eager_train_with_defun', MockIterator,
        resnet50_test_util.device_and_data_format(), defun=True)

  def benchmark_eager_train_datasets(self):
    # [FUNCTION DOC]
    # Benchmark: eager training fed from a repeating dataset (built on the CPU) through an
    # iterator, no compilation.

    def make_iterator(tensors):
      # [FUNCTION DOC]
      # Iterator factory: on the CPU device builds a dataset repeating the given tensors
      # forever and returns a Python iterator over it.
      with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_tensors(tensors).repeat()
      return iter(ds)

    self._benchmark_eager_train(
        'eager_train_dataset',
        make_iterator,
        resnet50_test_util.device_and_data_format(),
        defun=False)

  def benchmark_eager_train_datasets_with_defun(self):
    # [FUNCTION DOC]
    # Benchmark: training fed from a repeating dataset iterator with the model call and apply
    # step compiled.

    def make_iterator(tensors):
      # [FUNCTION DOC]
      # Iterator factory: on the CPU device builds a dataset repeating the given tensors
      # forever and returns a Python iterator over it.
      with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_tensors(tensors).repeat()
      return iter(ds)

    self._benchmark_eager_train(
        'eager_train_dataset_with_defun', make_iterator,
        resnet50_test_util.device_and_data_format(), defun=True)


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
