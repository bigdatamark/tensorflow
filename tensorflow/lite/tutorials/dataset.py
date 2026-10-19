# [MODULE DOC]
# Tutorial helper module giving tf.data dataset access to MNIST (cloned from the official models
# repository). Provides byte reading, header validation of image/label files, download with
# decompression, and train/test dataset builders that yield (flattened 784 float image scaled to
# 0..1, int32 label) pairs. No import-time side effects.
#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""tf.data.Dataset interface to the MNIST dataset.

 This is cloned from
 https://github.com/tensorflow/models/blob/master/official/r1/mnist/dataset.py
"""

import gzip
import os
import shutil
import tempfile
import urllib

import numpy as np
import tensorflow as tf


def read32(bytestream):
  # [FUNCTION DOC]
  # HELPER: reads big-endian integers from MNIST file headers.
  # Reads 4 bytes from the given binary stream and returns them as an unsigned 32-bit big-endian
  # integer (numpy scalar). Advances the stream position.
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
  # [FUNCTION DOC]
  # HELPER: validates MNIST image files before building the dataset.
  # KNOWN ISSUE: uses a misspelled file class name (Gfile instead of GFile) in the TensorFlow
  # file-IO module, so this would fail with an attribute error when actually called.
  # Opens the file in binary mode, reads magic number, image count (ignored), rows and columns.
  # Raises a value error if the magic number is not the image magic or images are not 28x28.
  # Returns nothing.
  """Validate that filename corresponds to images for the MNIST dataset."""
  with tf.io.gfile.Gfile(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_images, unused
    rows = read32(f)
    cols = read32(f)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))
    if rows != 28 or cols != 28:
      raise ValueError(
          'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
          (f.name, rows, cols))


def check_labels_file_header(filename):
  # [FUNCTION DOC]
  # HELPER: validates MNIST label files before building the dataset.
  # KNOWN ISSUE: uses the removed TensorFlow 1.x file API for opening files, which does not exist
  # in the TensorFlow 2 top-level namespace, so it would fail with an attribute error.
  # Opens the file in binary mode, reads magic number and item count (ignored). Raises a value
  # error if the magic number is not the labels magic. Returns nothing.
  """Validate that filename corresponds to labels for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_items, unused
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))


def download(directory, filename):
  # [FUNCTION DOC]
  # HELPER: fetches MNIST data files for the dataset builders.
  # KNOWN ISSUE: writes with a misspelled file class (Gfile instead of GFile), and only imports
  # the top-level urllib package, so the request helper may be unavailable; the download path
  # can fail at runtime.
  # Inputs: directory (target folder, created if absent) and filename (file name without the
  # compressed suffix). If the file already exists it is returned immediately. Otherwise downloads
  # the gzip version from a public mirror to a temporary file (prints a message), decompresses it
  # into the target path, deletes the temporary file, and returns the target path.
  """Download (and unzip) a file from the MNIST dataset if not already done."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
  _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
  print('Downloading %s to %s' % (url, zipped_filepath))
  urllib.request.urlretrieve(url, zipped_filepath)
  with gzip.open(zipped_filepath, 'rb') as f_in, tf.io.gfile.Gfile(
      filepath, 'wb'
  ) as f_out:
    shutil.copyfileobj(f_in, f_out)
  os.remove(zipped_filepath)
  return filepath


def dataset(directory, images_file, labels_file):
  # [FUNCTION DOC]
  # Builds a tf.data dataset of (image, label) pairs from MNIST files.
  # Inputs: directory for the data, plus names of the images file and labels file. Downloads both
  # if needed, validates their headers, then reads fixed-length records (skipping 16 and 8 header
  # bytes) and zips them. Returns the zipped dataset: images are flat 784 floats scaled to 0..1,
  # labels are int32 scalars. Raises a value error on invalid files.
  """Download and parse MNIST dataset."""

  images_file = download(directory, images_file)
  labels_file = download(directory, labels_file)

  check_image_file_header(images_file)
  check_labels_file_header(labels_file)

  def decode_image(image):
    # [FUNCTION DOC]
    # HELPER: dataset map function for images.
    # Decodes a raw byte record into unsigned bytes, casts to float, reshapes to a flat vector of
    # 784 values and divides by 255 so values lie in 0..1. Returns that float tensor.
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.io.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0

  def decode_label(label):
    # [FUNCTION DOC]
    # HELPER: dataset map function for labels.
    # Decodes a one-byte raw record into an unsigned byte, reshapes to a scalar and returns it
    # cast to a 32-bit integer tensor.
    label = tf.io.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.cast(label, tf.int32)

  images = tf.data.FixedLengthRecordDataset(
      images_file, 28 * 28, header_bytes=16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
  return tf.data.Dataset.zip((images, labels))


def train(directory):
  # [FUNCTION DOC]
  # Returns the MNIST training dataset (60000 examples) stored in the given directory, downloading
  # the standard train image and label files if missing. See the dataset builder for details.
  """tf.data.Dataset object for MNIST training data."""
  return dataset(directory, 'train-images-idx3-ubyte',
                 'train-labels-idx1-ubyte')


def test(directory):
  # [FUNCTION DOC]
  # Returns the MNIST test dataset (10000 examples) stored in the given directory, downloading
  # the standard test image and label files if missing. See the dataset builder for details.
  """tf.data.Dataset object for MNIST test data."""
  return dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
