# [MODULE DOC]
# HELPER: shared helper module for the Atheris Python fuzz targets in this directory.
# Defines the FuzzingHelper class, a thin wrapper over the Atheris fuzzed data provider that
# consumes booleans, bounded integers and floats, lists, strings, TensorFlow dtypes and random
# numeric tensors from the raw fuzzer bytes. Also holds module-private default bounds (integer,
# float, length, tensor size capped at 8 to avoid out-of-memory) and the lists of all dtypes and
# of dtypes supported by random-uniform generation. Importing requires Atheris and TensorFlow;
# there are no other import-time side effects.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Helper class for TF Python fuzzing."""

import atheris
import tensorflow as tf

_MIN_INT = -10000
_MAX_INT = 10000

_MIN_FLOAT = -10000.0
_MAX_FLOAT = 10000.0

_MIN_LENGTH = 0
_MAX_LENGTH = 10000

# Max shape can be 8 in length and randomized from 0-8 without running into an
# OOM error.
_MIN_SIZE = 0
_MAX_SIZE = 8

_TF_DTYPES = [
    tf.half, tf.float16, tf.float32, tf.float64, tf.bfloat16, tf.complex64,
    tf.complex128, tf.int8, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int16,
    tf.int32, tf.int64, tf.bool, tf.string, tf.qint8, tf.quint8, tf.qint16,
    tf.quint16, tf.qint32, tf.resource, tf.variant
]

# All types supported by tf.random.uniform
_TF_RANDOM_DTYPES = [tf.float16, tf.float32, tf.float64, tf.int32, tf.int64]


class FuzzingHelper(object):
  # [CLASS DOC]
  # HELPER: shared helper for the Atheris Python fuzz targets.
  # Wraps the Atheris fuzzed data provider built from a raw byte string and offers typed consumers:
  # booleans, bounded integers and floats, integer/float lists, strings, TensorFlow dtypes and
  # random numeric tensors. State: a single attribute holding the data provider; every call
  # consumes bytes from it, so results are deterministic for a given input and run out into default
  # values when the bytes are exhausted. Created once per fuzz iteration by the fuzz entry function.
  """FuzzingHelper makes handling FuzzedDataProvider easier with TensorFlow Python fuzzing."""

  def __init__(self, input_bytes):
    # [FUNCTION DOC]
    # Constructor. Input: input_bytes, the raw fuzzer byte string. Creates the Atheris fuzzed data
    # provider over those bytes and stores it in an attribute. Returns nothing; no other side effects.
    """FuzzingHelper initializer.

    Args:
      input_bytes: Input randomized bytes used to create a FuzzedDataProvider.
    """
    self.fdp = atheris.FuzzedDataProvider(input_bytes)

  def get_bool(self):
    # [FUNCTION DOC]
    # Consumes bytes from the provider to produce one boolean. No inputs. Returns the boolean
    # (false-like defaults once bytes run out). Side effect: advances the provider position.
    """Consume a bool.

    Returns:
      Consumed a bool based on input bytes and constraints.
    """
    return self.fdp.ConsumeBool()

  def get_int(self, min_int=_MIN_INT, max_int=_MAX_INT):
    # [FUNCTION DOC]
    # Consumes bytes to produce one integer within an inclusive range. Inputs: min_int and max_int,
    # the inclusive bounds (default to the module bounds of minus and plus ten thousand). Returns the
    # integer. Side effect: advances the provider position. Bound errors are left to the provider.
    """Consume a signed integer with given constraints.

    Args:
      min_int: Minimum allowed integer.
      max_int: Maximum allowed integer.

    Returns:
      Consumed integer based on input bytes and constraints.
    """
    return self.fdp.ConsumeIntInRange(min_int, max_int)

  def get_float(self, min_float=_MIN_FLOAT, max_float=_MAX_FLOAT):
    # [FUNCTION DOC]
    # Consumes bytes to produce one floating-point number within a range. Inputs: min_float and
    # max_float, the bounds (default minus and plus ten thousand). Returns the float. Side effect:
    # advances the provider position.
    """Consume a float with given constraints.

    Args:
      min_float: Minimum allowed float.
      max_float: Maximum allowed float.

    Returns:
      Consumed float based on input bytes and constraints.
    """
    return self.fdp.ConsumeFloatInRange(min_float, max_float)

  def get_int_list(self,
                   min_length=_MIN_LENGTH,
                   max_length=_MAX_LENGTH,
                   min_int=_MIN_INT,
                   max_int=_MAX_INT):
    # [FUNCTION DOC]
    # Consumes a length and then that many integers. Inputs: min_length and max_length, inclusive
    # bounds of the list length (defaults zero and ten thousand); min_int and max_int, inclusive
    # bounds of each element (defaults minus and plus ten thousand). Returns the list of integers.
    # Side effect: advances the provider position.
    """Consume a signed integer list with given constraints.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.
      min_int: Minimum allowed integer.
      max_int: Maximum allowed integer.

    Returns:
      Consumed integer list based on input bytes and constraints.
    """
    length = self.get_int(min_length, max_length)
    return self.fdp.ConsumeIntListInRange(length, min_int, max_int)

  def get_float_list(self, min_length=_MIN_LENGTH, max_length=_MAX_LENGTH):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns an integer list, but it returns a list of floats.
    # Consumes a length and then that many floats, each within the fixed module bounds (minus and
    # plus ten thousand; the element range cannot be changed). Inputs: min_length and max_length,
    # inclusive bounds of the list length (defaults zero and ten thousand). Returns the list of
    # floats. Side effect: advances the provider position.
    """Consume a float list with given constraints.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.

    Returns:
      Consumed integer list based on input bytes and constraints.
    """
    length = self.get_int(min_length, max_length)
    return self.fdp.ConsumeFloatListInRange(length, _MIN_FLOAT, _MAX_FLOAT)

  def get_int_or_float_list(self,
                            min_length=_MIN_LENGTH,
                            max_length=_MAX_LENGTH):
    # [FUNCTION DOC]
    # Consumes a bool and then returns either an integer list or a float list. Inputs: min_length
    # and max_length, bounds on the list length (defaults zero and ten thousand). Returns a list of
    # integers (if the bool is true) or of floats (otherwise), built by the corresponding list
    # method. Side effect: advances the provider position.
    """Consume a signed integer or float list with given constraints based on a consumed bool.

    Args:
      min_length: The minimum length of the list.
      max_length: The maximum length of the list.

    Returns:
      Consumed integer or float list based on input bytes and constraints.
    """
    if self.get_bool():
      return self.get_int_list(min_length, max_length)
    else:
      return self.get_float_list(min_length, max_length)

  def get_tf_dtype(self, allowed_set=None):
    # [FUNCTION DOC]
    # Picks a TensorFlow dtype using a consumed integer as an index. Input: allowed_set, optional
    # list of dtypes to choose from; when omitted or empty the full module list of dtypes (half,
    # floats, complex, integers, bool, string, quantized, resource, variant) is used. Returns the
    # chosen dtype. Raises an invalid-argument error if the chosen entry of allowed_set is not in
    # the module list of known dtypes (note only the picked entry is checked, so other entries may
    # be invalid). Side effect: advances the provider position.
    """Return a random tensorflow dtype.

    Args:
      allowed_set: An allowlisted set of dtypes to choose from instead of all of
      them.

    Returns:
      A random type from the list containing all TensorFlow types.
    """
    if allowed_set:
      index = self.get_int(0, len(allowed_set) - 1)
      if allowed_set[index] not in _TF_DTYPES:
        raise tf.errors.InvalidArgumentError(
            None, None,
            'Given dtype {} is not accepted.'.format(allowed_set[index]))
      return allowed_set[index]
    else:
      index = self.get_int(0, len(_TF_DTYPES) - 1)
      return _TF_DTYPES[index]

  def get_string(self, byte_count=_MAX_INT):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the result depends on a consumed bool, but no bool is
    # consumed; the string is simply read from the provider with the given maximum size.
    # Consumes up to byte_count bytes (default ten thousand) and decodes them into a text string.
    # Input: byte_count, the maximum number of bytes to consume. Returns the string. Side effect:
    # advances the provider position.
    """Consume a string with given constraints based on a consumed bool.

    Args:
      byte_count: Byte count that defaults to _MAX_INT.

    Returns:
      Consumed string based on input bytes and constraints.
    """
    return self.fdp.ConsumeString(byte_count)

  def get_random_numeric_tensor(self,
                                dtype=None,
                                min_size=_MIN_SIZE,
                                max_size=_MAX_SIZE,
                                min_val=_MIN_INT,
                                max_val=_MAX_INT):
    # [FUNCTION DOC]
    # Builds a random tensor from fuzzer bytes. Inputs: dtype, optional; must be one of the types
    # random-uniform supports (half, single, double precision float, 32 or 64 bit integer), and when
    # omitted one is picked at random; min_size and max_size, bounds for both the number of
    # dimensions and the size of each dimension (defaults zero and eight); min_val and max_val, the
    # value range (defaults minus and plus ten thousand). Draws a seed and a shape list, then
    # returns a uniformly random tensor of that shape and dtype. Raises an invalid-argument error
    # if max_size exceeds eight (out-of-memory guard) or if dtype is unsupported. Side effect:
    # advances the provider position.
    """Return a tensor of random shape and values.

    Generated tensors are capped at dimension sizes of 8, as 2^32 bytes of
    requested memory crashes the fuzzer (see b/34190148).
    Returns only type that tf.random.uniform can generate. If you need a
    different type, consider using tf.cast.

    Args:
      dtype: Type of tensor, must of one of the following types: float16,
        float32, float64, int32, or int64
      min_size: Minimum size of returned tensor
      max_size: Maximum size of returned tensor
      min_val: Minimum value in returned tensor
      max_val: Maximum value in returned tensor

    Returns:
      Tensor of random shape filled with uniformly random numeric values.
    """
    # Max shape can be 8 in length and randomized from 0-8 without running into
    # an OOM error.
    if max_size > 8:
      raise tf.errors.InvalidArgumentError(
          None, None,
          'Given size of {} will result in an OOM error'.format(max_size))

    seed = self.get_int()
    shape = self.get_int_list(
        min_length=min_size,
        max_length=max_size,
        min_int=min_size,
        max_int=max_size)

    if dtype is None:
      dtype = self.get_tf_dtype(allowed_set=_TF_RANDOM_DTYPES)
    elif dtype not in _TF_RANDOM_DTYPES:
      raise tf.errors.InvalidArgumentError(
          None, None,
          'Given dtype {} is not accepted in get_random_numeric_tensor'.format(
              dtype))

    return tf.random.uniform(
        shape=shape, minval=min_val, maxval=max_val, dtype=dtype, seed=seed)
