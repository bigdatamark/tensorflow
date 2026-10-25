# [MODULE DOC]
# HELPER: shared utilities for the stateless and stateful random op modules.
# Defines the public enumeration of random number generation algorithms (exported as the
# tf.random Algorithm type) and helpers that convert an algorithm given as an integer, enum,
# tensor or string into its integer code, and that derive the key and counter inputs expected
# by the raw V2 stateless random ops from a two-element seed. Import-time side effect: only the
# export registration of the Algorithm enumeration.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for random ops to share common usages."""

import enum

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("random.Algorithm", "random.experimental.Algorithm")
class Algorithm(enum.Enum):
  # [CLASS DOC]
  # Enumeration of supported random number generation algorithms: Philox (1), ThreeFry (2) and
  # automatic selection by device (3). The integer values must match the C++ framework's
  # algorithm constants. Members are used as the algorithm argument of generators and stateless
  # random ops; the class is exported publicly under two names.
  """A random-number-generation (RNG) algorithm.

  Many random-number generators (e.g. the `alg` argument of
  `tf.random.Generator` and `tf.random.stateless_uniform`) in TF allow
  you to choose the algorithm used to generate the (pseudo-)random
  numbers. You can set the algorithm to be one of the options below.

  * `PHILOX`: The Philox algorithm introduced in the paper ["Parallel
    Random Numbers: As Easy as 1, 2,
    3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
  * `THREEFRY`: The ThreeFry algorithm introduced in the paper
    ["Parallel Random Numbers: As Easy as 1, 2,
    3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
  * `AUTO_SELECT`: Allow TF to automatically select the algorithm
    depending on the accelerator device. Note that with this option,
    running the same TF program on different devices may result in
    different random numbers. Also note that TF may select an
    algorithm that is different from `PHILOX` and `THREEFRY`.
  """

  # The numbers here must match framework/rng_alg.h
  PHILOX = 1
  THREEFRY = 2
  AUTO_SELECT = 3


def convert_alg_to_int(alg):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the integer check comes first, so a boolean is treated as an integer; also an
  # Algorithm member or integer is not validated against supported values here.
  # Normalizes an algorithm specification to its integer code. Input alg may be an integer
  # (returned unchanged), an Algorithm member (its value), a tensor (returned unchanged) or a
  # string (case-insensitive; dashes, underscores and surrounding whitespace ignored) naming
  # philox, threefry or autoselect. Raises a value error for an unknown string and a type error
  # for any other type. No side effects.
  """Converts algorithm to an integer.

  Args:
    alg: can be one of these types: integer, Algorithm, Tensor, string. Allowed
      strings are "philox" and "threefry".

  Returns:
    An integer, unless the input is a Tensor in which case a Tensor is returned.
  """
  if isinstance(alg, int):
    return alg
  if isinstance(alg, Algorithm):
    return alg.value
  if isinstance(alg, tensor.Tensor):
    return alg
  if isinstance(alg, str):
    # canonicalized alg
    canon_alg = alg.strip().lower().replace("-", "").replace("_", "")
    if canon_alg == "philox":
      return Algorithm.PHILOX.value
    elif canon_alg == "threefry":
      return Algorithm.THREEFRY.value
    elif canon_alg == "autoselect":
      return Algorithm.AUTO_SELECT.value
    else:
      raise ValueError(unsupported_alg_error_msg(alg))
  else:
    raise TypeError(
        f"Can't convert argument `alg` (of value {alg} and type {type(alg)}) "
        "to int."
    )


def _get_key_counter(seed, alg):
  # [FUNCTION DOC]
  # HELPER: computes the key and counter inputs for raw V2 stateless random ops.
  # Inputs: seed (integer tensor of two elements) and alg (integer algorithm code). For
  # auto-select, asks the raw op to choose key and counter at run time by device; for Philox,
  # scrambles the seed with the Philox helper; for ThreeFry, packs the two seed words (cast to
  # 32-bit) into one 64-bit key and uses a zero counter. Returns a pair of key and counter
  # tensors. Raises a value error for an unsupported algorithm code.
  """Calculates the key and counter to pass to raw RNG ops.

  This function calculates the key and counter that will be passed to
  the raw RNG ops like `StatelessRandomUniformV2`. Depending on the
  input `alg`, the key and counter may be scrambled or copied from
  `seed`. If `alg` is `"auto_select"`, the key and counter will be
  determined at runtime based on device type.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.
    alg: The RNG algorithm. See `tf.random.stateless_uniform` for an
      explanation.

  Returns:
    A pair (key, counter) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
  if alg == Algorithm.AUTO_SELECT.value:
    key, counter = gen_stateless_random_ops_v2.stateless_random_get_key_counter(
        seed
    )
  elif alg == Algorithm.PHILOX.value:
    key, counter = _philox_scramble_seed(seed)
  elif alg == Algorithm.THREEFRY.value:
    key = array_ops.reshape(
        _uint32s_to_uint64(math_ops.cast(seed, dtypes.uint32)), [1]
    )
    counter = array_ops.zeros([1], dtypes.uint64)
  else:
    raise ValueError(unsupported_alg_error_msg(alg))
  return key, counter


def get_key_counter_alg(seed, alg):
  # [FUNCTION DOC]
  # HELPER: prepares all three algorithm-related inputs for raw V2 stateless random ops.
  # Inputs: seed (integer tensor of two elements) and alg (any form accepted by the algorithm
  # converter; nothing means auto-select). Converts alg to its integer code, derives key and
  # counter, and returns a triple of key, counter and integer algorithm. Raises errors from the
  # converter or key derivation for unsupported algorithms.
  """Calculates the key, counter and algorithm to pass to raw RNG ops.

  This function calculates the key and counter, and determines the algorithm
  that will be passed to the raw RNG ops like `StatelessRandomUniformV2`.
  Depending on the input `alg`, the key and counter may be scrambled or copied
  from `seed`. If `alg` is `"auto_select"`, the key and counter will be
  determined at runtime based on device type.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.
    alg: The RNG algorithm. See `tf.random.stateless_uniform` for an
      explanation.

  Returns:
    A pair (key, counter, algorithm) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
  if alg is None:
    alg = Algorithm.AUTO_SELECT.value
  alg = convert_alg_to_int(alg)
  key, counter = _get_key_counter(seed, alg)
  return key, counter, alg


def _uint32s_to_uint64(x):
  # [FUNCTION DOC]
  # HELPER: packs two 32-bit words into one 64-bit value.
  # Input x is a tensor whose first two elements are the low and high words. Casts each to
  # unsigned 64-bit, shifts the second left by 32 bits and combines them with bitwise or.
  # Returns the scalar unsigned 64-bit tensor.
  return bitwise_ops.bitwise_or(
      math_ops.cast(x[0], dtypes.uint64),
      bitwise_ops.left_shift(
          math_ops.cast(x[1], dtypes.uint64),
          constant_op.constant(32, dtypes.uint64),
      ),
  )


def unsupported_alg_error_msg(alg):
  # [FUNCTION DOC]
  # HELPER: builds the error text used when an algorithm value is not supported.
  # Input alg is the offending value. The message lists the supported values for Philox,
  # ThreeFry and auto-select, formatted as integers if alg is an integer, as lowercase names if
  # it is a string, and as enumeration members otherwise. Returns the message string; raises
  # nothing.
  """Produces the unsupported-algorithm error message."""
  if isinstance(alg, int):
    philox = Algorithm.PHILOX.value
    threefry = Algorithm.THREEFRY.value
    auto_select = Algorithm.AUTO_SELECT.value
  elif isinstance(alg, str):
    philox = "philox"
    threefry = "threefry"
    auto_select = "auto_select"
  else:
    philox = Algorithm.PHILOX
    threefry = Algorithm.THREEFRY
    auto_select = Algorithm.AUTO_SELECT
  return (
      f"Argument `alg` got unsupported value {alg}. Supported values are "
      f"{philox} for the Philox algorithm, "
      f"{threefry} for the ThreeFry algorithm, and "
      f"{auto_select} for auto-selection."
  )


def _philox_scramble_seed(seed):
  # [FUNCTION DOC]
  # HELPER: seed scrambling for the Philox algorithm.
  # Input seed is an integer tensor of two elements. Uses a fixed constant key and the seed as
  # counter to generate four 32-bit random words with the raw full-integer stateless op, then
  # packs the first two words into a one-element 64-bit key and builds a two-element counter
  # from zero and the packed last two words. Mirrors the C++ stateless kernel scrambling.
  # Returns a pair of key and counter tensors.
  """Determines the key and counter for Philox PRNG with the given seed.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.

  Returns:
    A pair (key, counter) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
  # the same scrambling procedure as core/kernels/stateless_random_ops.cc
  key = constant_op.constant([0x02461E293EC8F720], dtypes.uint64)
  counter = math_ops.cast(seed, dtypes.uint64)
  mix = gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(
      [4],
      key=key,
      counter=counter,
      dtype=dtypes.uint32,
      alg=Algorithm.PHILOX.value,
  )
  key = array_ops.reshape(_uint32s_to_uint64(mix[:2]), [1])
  counter = array_ops_stack.stack([0, _uint32s_to_uint64(mix[2:])], axis=0)
  return key, counter
