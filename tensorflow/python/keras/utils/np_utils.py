# [MODULE DOC]
# Small numpy-only helpers used by Keras: one-hot encoding of integer class labels
# and normalization of arrays by their norm along an axis. Exposes two public functions,
# to_categorical and normalize. No import-time side effects other than importing numpy.
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
"""Numpy-related utilities."""

import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring says classes range from 0 to num_classes inclusive and mentions a
  # "Value Error" for string input; actually labels must be 0..num_classes-1 (a label equal to
  # num_classes causes an index error from numpy), and string input fails in the integer
  # conversion with whatever error numpy raises, not an explicit check. Also, a num_classes of
  # zero is treated like nothing and inferred from the data.
  # One-hot encodes integer class labels. Input y is any array-like of non-negative integers;
  # it is converted to an integer array. A trailing axis of length one (when there is more than
  # one axis) is dropped from the output shape. num_classes is optional: if null or zero it is
  # inferred as the largest label plus one. dtype is the numpy element type of the result
  # (default float32). Returns a numpy array shaped like the (squeezed) input with an extra
  # last axis of size num_classes, holding one at each label position and zero elsewhere.
  # Raises numpy errors for non-integer-convertible input or labels outside the class range.
  # No side effects; the input is not modified.
  """Converts a class vector (integers) to binary class matrix.

  E.g. for use with categorical_crossentropy.

  Args:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes. If `None`, this would be inferred
        as the (largest number in `y`) + 1.
      dtype: The data type expected by the input. Default: `'float32'`.

  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.

  Example:

  >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
  >>> a = tf.constant(a, shape=[4, 4])
  >>> print(a)
  tf.Tensor(
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)

  >>> b = tf.constant([.9, .04, .03, .03,
  ...                  .3, .45, .15, .13,
  ...                  .04, .01, .94, .05,
  ...                  .12, .21, .5, .17],
  ...                 shape=[4, 4])
  >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
  >>> print(np.around(loss, 5))
  [0.10536 0.82807 0.1011  1.77196]

  >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
  >>> print(np.around(loss, 5))
  [0. 0. 0. 0.]

  Raises:
      Value Error: If input contains string value

  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


def normalize(x, axis=-1, order=2):
  # [FUNCTION DOC]
  # Divides a numpy array by its vector norm computed along one axis. x is the array, axis is
  # the axis to reduce (default last), order is the norm order (default 2, the Euclidean norm).
  # Entries whose norm is zero are replaced by one so those slices are left unchanged rather
  # than producing division by zero. Returns a new array of the same shape; the input is not
  # modified. Errors come from numpy if the axis or order is invalid.
  """Normalizes a Numpy array.

  Args:
      x: Numpy array to normalize.
      axis: axis along which to normalize.
      order: Normalization order (e.g. `order=2` for L2 norm).

  Returns:
      A normalized copy of the array.
  """
  l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
  l2[l2 == 0] = 1
  return x / np.expand_dims(l2, axis)
