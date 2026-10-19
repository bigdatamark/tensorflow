# [MODULE DOC]
# HELPER: shared utility module for the eager ResNet50 tests and benchmarks in this directory.
# Provides a device and data-layout chooser, a synthetic image and one-hot label batch
# generator, and a benchmark result reporter. No import-time side effects.
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

import time

import tensorflow as tf


def device_and_data_format():
  # [FUNCTION DOC]
  # HELPER: device selection for the eager ResNet50 tests and benchmarks.
  # Returns a pair of strings: the first GPU device name with channels-first layout when a
  # physical GPU exists, otherwise the CPU device name with channels-last layout.
  if tf.config.list_physical_devices('GPU'):
    return ('/gpu:0', 'channels_first')
  return ('/cpu:0', 'channels_last')


def random_batch(batch_size, data_format, seed=None):
  # [FUNCTION DOC]
  # HELPER: synthetic data generator for the eager ResNet50 tests and benchmarks.
  # Inputs: batch_size, number of examples; data_format, channels-first or channels-last (any
  # other string is treated as channels-last); seed, optional random seed (a falsy seed such as
  # zero or nothing is ignored). Returns a pair of tensors: uniform random images of shape batch
  # by 224 by 224 by 3 (or 3 by 224 by 224 for channels-first), and one-hot labels of shape batch
  # by 1000 from random class ids. Side effect: sets the global random seed when a seed is given.
  """Create synthetic resnet50 images and labels for testing."""
  if seed:
    tf.random.set_seed(seed)
  shape = (3, 224, 224) if data_format == 'channels_first' else (224, 224, 3)
  shape = (batch_size,) + shape

  num_classes = 1000
  images = tf.random.uniform(shape)
  labels = tf.random.uniform([batch_size],
                             minval=0,
                             maxval=num_classes,
                             dtype=tf.int32)
  one_hot = tf.one_hot(labels, num_classes)

  return images, one_hot


def report(benchmark, label, start, num_iters, device, batch_size, data_format,
           num_replicas=1):
  # [FUNCTION DOC]
  # HELPER: benchmark result reporter for the eager ResNet50 tests and benchmarks.
  # Inputs: benchmark, the benchmark object to report through; label, workload name; start,
  # wall-clock start time of the timed loop; num_iters, number of timed iterations; device,
  # device name string (its device type is lower-cased for the report name); batch_size,
  # per-replica batch size; data_format, layout string; num_replicas, replica count (default
  # one, which omits the replica part of the name). Computes average time per iteration since
  # start and reports a name built from label, device type, batch size, replicas and layout,
  # with examples per second computed over all replicas. Returns nothing.
  avg_time = (time.time() - start) / num_iters
  dev = tf.DeviceSpec.from_string(device).device_type.lower()
  replica_str = '' if num_replicas == 1 else 'replicas_%d_' % num_replicas
  name = '%s_%s_batch_%d_%s%s' % (label, dev, batch_size,
                                  replica_str, data_format)
  extras = {'examples_per_sec': (num_replicas * batch_size) / avg_time}
  benchmark.report_benchmark(
      iters=num_iters, wall_time=avg_time, name=name, extras=extras)
