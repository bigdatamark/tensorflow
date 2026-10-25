# [MODULE DOC]
# HELPER: Test utility module for the profiler integration tests.
# Provides a synthetic MNIST-like dataset builder and a small deterministically initialized
# convolutional Keras model used to exercise profiling under distribution strategies.
# No import-time side effects besides importing TensorFlow.
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
"""A simple MNIST model for testing multi-worker distribution strategies with Keras."""

import tensorflow as tf


def mnist_synthetic_dataset(batch_size, steps_per_epoch):
  # [FUNCTION DOC]
  # HELPER: Supplies data for the profiler integration tests.
  # KNOWN ISSUE: The training dataset is always batched with a fixed size of 64; the batch_size
  # argument only affects how many all-ones samples are created, not the batch size.
  # Inputs: batch_size and steps_per_epoch (integers; their product is the number of training
  # samples). Builds a training dataset of all-one images (28 by 28 by 1) with all-one integer
  # labels, repeated forever and batched by 64 dropping the remainder, and an evaluation dataset
  # of 10000 random images with random labels from 0 to 8, batched by 64.
  # Returns the pair (training dataset, evaluation dataset). Uses random numbers for eval data.
  """Generate synthetic MNIST dataset for testing."""
  # train dataset
  x_train = tf.ones([batch_size * steps_per_epoch, 28, 28, 1],
                    dtype=tf.dtypes.float32)
  y_train = tf.ones([batch_size * steps_per_epoch, 1], dtype=tf.dtypes.int32)
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_ds = train_ds.repeat()
  # train_ds = train_ds.shuffle(100)
  train_ds = train_ds.batch(64, drop_remainder=True)

  # eval dataset
  x_test = tf.random.uniform([10000, 28, 28, 1], dtype=tf.dtypes.float32)
  y_test = tf.random.uniform([10000, 1],
                             minval=0,
                             maxval=9,
                             dtype=tf.dtypes.int32)
  eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  eval_ds = eval_ds.batch(64, drop_remainder=True)

  return train_ds, eval_ds


def get_mnist_model(input_shape):
  # [FUNCTION DOC]
  # HELPER: Model factory for the profiler integration tests.
  # Input: input_shape, the shape of one input image (height, width, channels). Builds a
  # functional Keras model: a 32-filter 3 by 3 convolution with relu, batch normalization,
  # flatten added to itself, and a 10-way softmax dense layer, with fixed-seed truncated normal
  # kernel initializers for determinism. Compiles it with sparse categorical cross-entropy,
  # stochastic gradient descent at learning rate 0.001 and an accuracy metric.
  # Returns the compiled model. Creates variables in the current strategy scope if any.
  """Define a deterministically-initialized CNN model for MNIST testing."""
  inputs = tf.keras.Input(shape=input_shape)
  x = tf.keras.layers.Conv2D(
      32,
      kernel_size=(3, 3),
      activation="relu",
      kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=99))(
          inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Flatten()(x) + tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      10,
      activation="softmax",
      kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=99))(
          x)
  model = tf.keras.Model(inputs=inputs, outputs=x)

  # TODO(yuefengz): optimizer with slot variables doesn't work because of
  # optimizer's bug.
  # TODO(yuefengz): we should not allow non-v2 optimizer.
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=["accuracy"])
  return model
