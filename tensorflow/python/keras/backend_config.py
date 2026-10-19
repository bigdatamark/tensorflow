# [MODULE DOC]
# HELPER: small configuration module supporting the Keras backend and layers.
# Holds three module-level settings for the legacy Keras backend: default float type name,
# epsilon fuzz factor, and default image data format, each with a getter and a validating
# (except epsilon) setter. Exposes epsilon, set_epsilon, floatx, set_floatx,
# image_data_format and set_image_data_format. Imports only the dispatch utilities; no
# import-time side effects beyond initializing the globals (single precision, one ten-millionth,
# channels-last).
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
"""Keras backend config API."""

from tensorflow.python.util import dispatch

# The type of float to use throughout a session.
_FLOATX = 'float32'

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = 'channels_last'


@dispatch.add_dispatch_support
def epsilon():
  # [FUNCTION DOC]
  # Reports the current fuzz factor used to avoid division by zero and similar numeric issues.
  # Input: none.
  # Output: a float, one ten-millionth by default unless changed by set_epsilon.
  # Side effects: none; reads module state. Wrapped with the dispatch decorator so other
  # types can override it via the dispatch mechanism.
  """Returns the value of the fuzz factor used in numeric expressions.

  Returns:
      A float.

  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  """
  return _EPSILON


def set_epsilon(value):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says value is a float, but no validation is done; any object
  # (including non-numbers or non-positive values) is stored as is.
  # Sets the module-wide fuzz factor used in numeric expressions.
  # Input: value, the new epsilon (expected to be a small positive float).
  # Output: returns nothing.
  # Side effects: overwrites the module-level epsilon global, which epsilon then reports.
  """Sets the value of the fuzz factor used in numeric expressions.

  Args:
      value: float. New value of epsilon.

  Example:
  >>> tf.keras.backend.epsilon()
  1e-07
  >>> tf.keras.backend.set_epsilon(1e-5)
  >>> tf.keras.backend.epsilon()
  1e-05
   >>> tf.keras.backend.set_epsilon(1e-7)
  """
  global _EPSILON
  _EPSILON = value


def floatx():
  # [FUNCTION DOC]
  # Reports the current default floating point type name.
  # Input: none.
  # Output: a string such as the single precision name (the default).
  # Side effects: none; reads module state. Not wrapped with the dispatch decorator, unlike
  # epsilon and image_data_format.
  """Returns the default float type, as a string.

  E.g. `'float16'`, `'float32'`, `'float64'`.

  Returns:
      String, the current default float type.

  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  """
  return _FLOATX


def set_floatx(value):
  # [FUNCTION DOC]
  # Sets the module-wide default floating point type name.
  # Input: value, a string that must be one of the half, single or double precision names.
  # Output: returns nothing.
  # Raises: a value error naming the unknown type when value is not one of the three allowed
  # names.
  # Side effects: overwrites the module-level global holding the float type (stored as a
  # string), which floatx then reports.
  """Sets the default float type.

  Note: It is not recommended to set this to float16 for training, as this will
  likely cause numeric stability issues. Instead, mixed precision, which is
  using a mix of float16 and float32, can be used by calling
  `tf.keras.mixed_precision.set_global_policy('mixed_float16')`. See the
  [mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for details.

  Args:
      value: String; `'float16'`, `'float32'`, or `'float64'`.

  Example:
  >>> tf.keras.backend.floatx()
  'float32'
  >>> tf.keras.backend.set_floatx('float64')
  >>> tf.keras.backend.floatx()
  'float64'
  >>> tf.keras.backend.set_floatx('float32')

  Raises:
      ValueError: In case of invalid value.
  """
  global _FLOATX
  if value not in {'float16', 'float32', 'float64'}:
    raise ValueError('Unknown floatx type: ' + str(value))
  _FLOATX = str(value)


@dispatch.add_dispatch_support
def image_data_format():
  # [FUNCTION DOC]
  # Reports the current default image data format convention.
  # Input: none.
  # Output: a string, either the channels-last or channels-first name (channels-last by
  # default unless changed by set_image_data_format).
  # Side effects: none; reads module state. Wrapped with the dispatch decorator so other
  # types can override it via the dispatch mechanism.
  """Returns the default image data format convention.

  Returns:
      A string, either `'channels_first'` or `'channels_last'`

  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  """
  return _IMAGE_DATA_FORMAT


def set_image_data_format(data_format):
  # [FUNCTION DOC]
  # Sets the module-wide default image data format convention.
  # Input: data_format, a string that must be either the channels-last or the channels-first
  # name; anything else (including non-strings) is rejected.
  # Output: returns nothing.
  # Raises: a value error when data_format is not one of the two accepted names.
  # Side effects: overwrites the module-level global holding the image data format (stored as
  # a string), which image_data_format then reports.
  """Sets the value of the image data format convention.

  Args:
      data_format: string. `'channels_first'` or `'channels_last'`.

  Example:
  >>> tf.keras.backend.image_data_format()
  'channels_last'
  >>> tf.keras.backend.set_image_data_format('channels_first')
  >>> tf.keras.backend.image_data_format()
  'channels_first'
  >>> tf.keras.backend.set_image_data_format('channels_last')

  Raises:
      ValueError: In case of invalid `data_format` value.
  """
  global _IMAGE_DATA_FORMAT
  if data_format not in {'channels_last', 'channels_first'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  _IMAGE_DATA_FORMAT = str(data_format)
