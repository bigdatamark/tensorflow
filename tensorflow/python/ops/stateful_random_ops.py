# [MODULE DOC]
# Stateful random number generation built on counter-based algorithms (Philox, ThreeFry). Defines
# constants for seed/state sizes and types, helpers that turn seeds into 1-D integer RNG states,
# the public Generator class (state held in an int64 variable, advanced on each draw and made
# distribution-strategy aware), and functions to get or replace a process-wide global generator.
# Generator draws are implemented by advancing the state variable and calling the stateless v2
# random ops with the resulting key and counter. Import-time side effects: public API export
# registrations only; the global generator is created lazily on first request.
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
"""Operations for generating random numbers."""

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# A seed for random ops (stateful and stateless) will always be 1024
# bits, all of which will be sent to the C++ code. The actual C++
# implementation of some algorithms may only use a lower part of the bits.

UINT64_HALF_SPAN = 2**63
MAX_INT64 = UINT64_HALF_SPAN - 1
MIN_INT64 = -UINT64_HALF_SPAN
UINT64_SPAN = UINT64_HALF_SPAN * 2
# 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained in
# b/111604096 and cl/171681867), so I use signed int here. I choose int64
# instead of int32 here because `VarHandleOp` doesn't support int32 on GPU.
SEED_TYPE = "int64"
SEED_MIN = MIN_INT64
SEED_MAX = MAX_INT64
SEED_UINT_SPAN = UINT64_SPAN
SEED_TYPE_BITS = 64
SEED_BIT_MASK = 0xFFFFFFFFFFFFFFFF
SEED_SIZE = 16  # in units of SEED_TYPE


STATE_TYPE = SEED_TYPE
ALGORITHM_TYPE = STATE_TYPE


# The following sizes are all in unit of uint64.
PHILOX_KEY_SIZE = 1
THREEFRY_KEY_SIZE = 1
PHILOX_COUNTER_SIZE = 2
THREEFRY_COUNTER_SIZE = 1
PHILOX_STATE_SIZE = PHILOX_COUNTER_SIZE + PHILOX_KEY_SIZE
THREEFRY_STATE_SIZE = THREEFRY_COUNTER_SIZE + THREEFRY_KEY_SIZE


RNG_ALG_PHILOX = random_ops_util.Algorithm.PHILOX.value
RNG_ALG_THREEFRY = random_ops_util.Algorithm.THREEFRY.value


DEFAULT_ALGORITHM = RNG_ALG_PHILOX


def non_deterministic_ints(shape, dtype=dtypes.int64):
  # [FUNCTION DOC]
  # Produces integers from an operating-system provided source of randomness. Inputs: shape of the
  # result (integer tensor or list) and dtype (optional integer type, default int64). Returns a
  # tensor of that shape and type whose values differ on every execution. Errors come from the
  # underlying op (for example for unsupported dtypes). Used to seed generators non-deterministically.
  """Non-deterministically generates some integers.

  This op may use some OS-provided source of non-determinism (e.g. an RNG), so
  each execution will give different results.

  Args:
    shape: the shape of the result.
    dtype: (optional) the dtype of the result.

  Returns:
    a tensor whose element values are non-deterministically chosen.
  """
  return gen_stateful_random_ops.non_deterministic_ints(
      shape=shape, dtype=dtype)


def _uint_to_int(n):
  # [FUNCTION DOC]
  # HELPER: seed/state conversion, avoids overflow when building int64 tensors.
  # If n is a Python integer above the largest signed 64-bit value, subtracts 2 to the 64th power so
  # it wraps into the signed range; any other value (including negative ints, lists, tensors) is
  # returned unchanged. Pure function.
  if isinstance(n, int) and n > SEED_MAX:
    n = n - SEED_UINT_SPAN
  return n


def _make_1d_state(state_size, seed):
  # [FUNCTION DOC]
  # HELPER: builds RNG state vectors for create_rng_state and the generator constructors.
  # Inputs: state_size, the required length; seed, a Python integer (split into 64-bit chunks, low
  # chunk first, up to state_size chunks) or a 1-D sequence/tensor of integers. The seed is mapped
  # to signed values, cast to int64, flattened, truncated to state_size, and padded with zeros on
  # the LEFT when too short (so a short seed lands in the key part rather than the counter, avoiding
  # overlapping streams). Returns a 1-D int64 tensor of static shape [state_size]. Works with
  # unknown static seed length by using the dynamic shape. Raises whatever the cast/reshape raises
  # for non-integer or malformed seeds.
  """Makes a 1-D RNG state.

  Args:
    state_size: an integer.
    seed: an integer or 1-D tensor.

  Returns:
    a 1-D tensor of shape [state_size] and dtype STATE_TYPE.
  """
  if isinstance(seed, int):
    # chop the Python integer (infinite precision) into chunks of SEED_TYPE
    ls = []
    for _ in range(state_size):
      ls.append(seed & SEED_BIT_MASK)
      seed >>= SEED_TYPE_BITS
    seed = ls
  # to avoid overflow error from ops.convert_to_tensor
  seed = nest.map_structure(_uint_to_int, seed)
  seed = math_ops.cast(seed, STATE_TYPE)
  seed = array_ops.reshape(seed, [-1])
  seed = seed[0:state_size]
  # Padding with zeros on the *left* if too short. Padding on the right would
  # cause a small seed to be used as the "counter" while the "key" is always
  # zero (for counter-based RNG algorithms), because in the current memory
  # layout counter is stored before key. In such a situation two RNGs with
  # two different small seeds may generate overlapping outputs.
  seed_size = seed.shape[0]
  if seed_size is None:
    seed_size = array_ops.shape(seed)[0]
  padding_size = math_ops.maximum(state_size - seed_size, 0)
  padding = array_ops.zeros([padding_size], seed.dtype)
  # can't use `pad` because it doesn't support integer dtypes on GPU
  seed = array_ops.concat([padding, seed], axis=0)
  seed.set_shape([state_size])
  return seed


def _get_counter_size(alg):
  # [FUNCTION DOC]
  # HELPER: algorithm lookup for key/counter splitting.
  # KNOWN ISSUE: the error path calls an error-message function on the stateless random ops module,
  # which does not define it (it lives in the random ops util module), so an unknown algorithm
  # actually raises an attribute error instead of the documented ValueError.
  # Maps an algorithm id (integer) to the counter length in 64-bit words: Philox 2, ThreeFry 1, and
  # auto-select is treated as Philox (2). Raises ValueError with the unsupported-algorithm message
  # for any other id.
  if alg == random_ops_util.Algorithm.PHILOX.value:
    return PHILOX_COUNTER_SIZE
  elif alg == random_ops_util.Algorithm.THREEFRY.value:
    return THREEFRY_COUNTER_SIZE
  elif alg == random_ops_util.Algorithm.AUTO_SELECT.value:
    # For AUTO_SELECT, we'll manage the counter as if it's for Philox.
    return PHILOX_COUNTER_SIZE
  else:
    raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))


def _get_state_size(alg):
  # [FUNCTION DOC]
  # HELPER: algorithm lookup for state vector length.
  # KNOWN ISSUE: the error path calls an error-message function on the stateless random ops module,
  # which does not define it (it lives in the random ops util module), so an unknown algorithm
  # actually raises an attribute error instead of the documented ValueError.
  # Maps an algorithm id (integer) to the full state length in 64-bit words: Philox 3, ThreeFry 2,
  # and auto-select is treated as Philox (3). Raises ValueError with the unsupported-algorithm
  # message for any other id.
  if alg == random_ops_util.Algorithm.PHILOX.value:
    return PHILOX_STATE_SIZE
  elif alg == random_ops_util.Algorithm.THREEFRY.value:
    return THREEFRY_STATE_SIZE
  elif alg == random_ops_util.Algorithm.AUTO_SELECT.value:
    # For AUTO_SELECT, we'll manage the state as if it's for Philox.
    return PHILOX_STATE_SIZE
  else:
    raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))


def _check_state_shape(shape, alg):
  # [FUNCTION DOC]
  # HELPER: validation for generator construction.
  # Checks that a state shape is compatible with a one-element shape holding the state size for alg.
  # If alg is a symbolic tensor in graph mode the check is skipped and nothing happens. Returns
  # nothing; raises a ValueError (from shape compatibility assertion) on mismatch, or the
  # unsupported-algorithm error for unknown algorithms.
  if isinstance(alg, tensor.Tensor) and not context.executing_eagerly():
    return
  shape.assert_is_compatible_with([_get_state_size(int(alg))])


def _make_state_from_seed(seed, alg):
  # [FUNCTION DOC]
  # HELPER: thin glue for create_rng_state.
  # Builds a 1-D int64 state for the integer algorithm id alg from the given seed, using the
  # algorithm's state size. Returns the state tensor; errors propagate from the size lookup
  # (ValueError for unknown algorithm) and the seed conversion.
  return _make_1d_state(_get_state_size(alg), seed)


@tf_export("random.create_rng_state", "random.experimental.create_rng_state")
def create_rng_state(seed, alg):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a numpy array, but it returns a tensor (the result
  # of tensor casting and concatenation).
  # Public function building an RNG state from an integer or vector seed. alg may be a string name,
  # an Algorithm enum or an integer; it is converted to an integer id. Returns a 1-D int64 tensor
  # whose length depends on the algorithm (3 for Philox, 2 for ThreeFry). Short seeds are
  # left-padded with zeros. Raises ValueError for unsupported algorithms.
  """Creates a RNG state from an integer or a vector.

  Example:

  >>> tf.random.create_rng_state(
  ...     1234, "philox")
  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1234,    0,    0])>
  >>> tf.random.create_rng_state(
  ...     [12, 34], "threefry")
  <tf.Tensor: shape=(2,), dtype=int64, numpy=array([12, 34])>

  Args:
    seed: an integer or 1-D numpy array.
    alg: the RNG algorithm. Can be a string, an `Algorithm` or an integer.

  Returns:
    a 1-D numpy array whose size depends on the algorithm.
  """
  alg = random_ops_util.convert_alg_to_int(alg)
  return _make_state_from_seed(seed, alg)


def _shape_tensor(shape):
  # [FUNCTION DOC]
  # HELPER: shape normalization for random draws.
  # Converts a shape (list, tuple, or tensor) to an integer tensor named shape. An empty list or
  # tuple becomes an int64 tensor (otherwise the dtype is inferred, int32 or int64). Returns the
  # tensor; raises if the value cannot be converted.
  """Convert to an int32 or int64 tensor, defaulting to int64 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int64
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")


def _convert_to_state_tensor(t):
  # [FUNCTION DOC]
  # HELPER: converts states, keys and counters for Generator.
  # Maps Python integers beyond the signed 64-bit range into the signed range (recursively through
  # nested structures) and then casts the result to the state type (int64). Returns a tensor.
  # to avoid out-of-range error from ops.convert_to_tensor
  t = nest.map_structure(_uint_to_int, t)
  return math_ops.cast(t, STATE_TYPE)


def get_replica_id():
  # [FUNCTION DOC]
  # HELPER: replica identification for per-replica key derivation in Generator.
  # Returns the replica id within the sync group from the current distribution replica context, or
  # nothing/null value when there is no replica context (cross-replica or no strategy).
  rctx = distribute_lib.get_replica_context()
  if rctx is None:
    return None
  return rctx.replica_id_in_sync_group


@tf_export("random.Generator", "random.experimental.Generator")
class Generator(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # Stateful random number generator. Holds an integer state variable (non-trainable int64
  # variable) and an algorithm id (Philox or ThreeFry), plus the distribution strategy active at
  # construction (or none). Created via the constructor or the class methods from_state,
  # from_seed, from_non_deterministic_state, from_key_counter. Each sampling call (normal,
  # uniform, truncated_normal, binomial, ...) advances the counter in the state and runs a
  # stateless op on the resulting key and counter; under a strategy, the key is mixed with the
  # replica id so replicas get different streams. Trackable, so it can be checkpointed. Also
  # offers reset methods, skip, make_seeds and split.
  """Random-number generator.

  Example:

  Creating a generator from a seed:

  >>> g = tf.random.Generator.from_seed(1234)
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[ 0.9356609 ,  1.0854305 , -0.93788373],
         [-0.5061547 ,  1.3169702 ,  0.7137579 ]], dtype=float32)>

  Creating a generator from a non-deterministic state:

  >>> g = tf.random.Generator.from_non_deterministic_state()
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...>

  All the constructors allow explicitly choosing an Random-Number-Generation
  (RNG) algorithm. Supported algorithms are `"philox"` and `"threefry"`. For
  example:

  >>> g = tf.random.Generator.from_seed(123, alg="philox")
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[ 0.8673864 , -0.29899067, -0.9310337 ],
         [-1.5828488 ,  1.2481191 , -0.6770643 ]], dtype=float32)>

  CPU, GPU and TPU with the same algorithm and seed will generate the same
  integer random numbers. Float-point results (such as the output of `normal`)
  may have small numerical discrepancies between different devices.

  This class uses a `tf.Variable` to manage its internal state. Every time
  random numbers are generated, the state of the generator will change. For
  example:

  >>> g = tf.random.Generator.from_seed(1234)
  >>> g.state
  <tf.Variable ... numpy=array([1234,    0,    0])>
  >>> g.normal(shape=(2, 3))
  <...>
  >>> g.state
  <tf.Variable ... numpy=array([2770,    0,    0])>

  The shape of the state is algorithm-specific.

  There is also a global generator:

  >>> g = tf.random.get_global_generator()
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...>

  When creating a generator inside a `tf.distribute.Strategy` scope, each
  replica will get a different stream of random numbers.

  For example, in this code:

  ```
  strat = tf.distribute.MirroredStrategy(devices=["cpu:0", "cpu:1"])
  with strat.scope():
    g = tf.random.Generator.from_seed(1)
    def f():
      return g.normal([])
    results = strat.run(f).values
  ```

  `results[0]` and `results[1]` will have different values.

  If the generator is seeded (e.g. created via `Generator.from_seed`), the
  random numbers will be determined by the seed, even though different replicas
  get different numbers.  One can think of a random number generated on a
  replica as a hash of the replica ID and a "master" random number that may be
  common to all replicas. Hence, the whole system is still deterministic.

  (Note that the random numbers on different replicas are not correlated, even
  if they are deterministically determined by the same seed. They are not
  correlated in the sense that no matter what statistics one calculates on them,
  there won't be any discernable correlation.)

  Generators can be freely saved and restored using `tf.train.Checkpoint`. The
  checkpoint can be restored in a distribution strategy with a different number
  of replicas than the original strategy. If a replica ID is present in both the
  original and the new distribution strategy, its state will be properly
  restored (i.e. the random-number stream from the restored point will be the
  same as that from the saving point) unless the replicas have already diverged
  in their RNG call traces before saving (e.g. one replica has made one RNG call
  while another has made two RNG calls). We don't have such guarantee if the
  generator is saved in a strategy scope and restored outside of any strategy
  scope, or vice versa.

  When a generator is created within the scope of
  `tf.distribute.experimental.ParameterServerStrategy`, the workers
  will share the generator's state (placed on one of the parameter
  servers). In this way the workers will still get different
  random-number streams, as stated above. (This is similar to replicas
  in a `tf.distribute.MirroredStrategy` sequentially accessing a
  generator created outside the strategy.) Each RNG call on a worker
  will incur a round-trip to a parameter server, which may have
  performance impacts. When creating a
  `tf.distribute.experimental.ParameterServerStrategy`, please make
  sure that the `variable_partitioner` argument won't shard small
  variables of shape `[2]` or `[3]` (because generator states must not
  be sharded). Ways to avoid sharding small variables include setting
  `variable_partitioner` to `None` or to
  `tf.distribute.experimental.partitioners.MinSizePartitioner` with a
  large enough `min_shard_bytes` (see
  `tf.distribute.experimental.ParameterServerStrategy`'s documentation
  for more details).
  """

  @classmethod
  def from_state(cls, state, alg):
    # [FUNCTION DOC]
    # Class method constructor. Builds a generator directly from a state (vector, tensor or
    # existing variable, which is then reused) and an algorithm (name, enum or integer). Returns
    # the new generator; errors come from the constructor (shape mismatch, unknown algorithm).
    """Creates a generator from a state.

    See `__init__` for description of `state` and `alg`.

    Args:
      state: the new state.
      alg: the RNG algorithm.

    Returns:
      The new generator.
    """
    return cls(alg=alg, state=state)

  @classmethod
  def from_seed(cls, seed, alg=None):
    # [FUNCTION DOC]
    # Class method constructor from an integer or vector seed. alg is optional; when omitted the
    # default (Philox) is used (the docstring's "auto-selected" is just this fixed default). The
    # algorithm is converted to an integer id, the seed converted to a state, and a new generator
    # with a fresh state variable is returned. Raises ValueError for unsupported algorithms.
    """Creates a generator from a seed.

    A seed is a 1024-bit unsigned integer represented either as a Python
    integer or a vector of integers. Seeds shorter than 1024-bit will be
    padded. The padding, the internal structure of a seed and the way a seed
    is converted to a state are all opaque (unspecified). The only semantics
    specification of seeds is that two different seeds are likely to produce
    two independent generators (but no guarantee).

    Args:
      seed: the seed for the RNG.
      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    if alg is None:
      # TODO(b/170668986): more sophisticated algorithm selection
      alg = DEFAULT_ALGORITHM
    alg = random_ops_util.convert_alg_to_int(alg)
    state = create_rng_state(seed, alg)
    return cls(state=state, alg=alg)

  @classmethod
  def from_non_deterministic_state(cls, alg=None):
    # [FUNCTION DOC]
    # Class method constructor that fills the state with operating-system randomness. alg is
    # optional (default Philox). Raises RuntimeError if op determinism is enabled. Otherwise
    # draws a state of the algorithm's size via the non-deterministic integer op and returns a new
    # generator. Raises ValueError for unsupported algorithms.
    """Creates a generator by non-deterministically initializing its state.

    The source of the non-determinism will be platform- and time-dependent.

    Args:
      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    if config.is_op_determinism_enabled():
      raise RuntimeError('"from_non_deterministic_state" cannot be called when '  # pylint: disable=g-doc-exception
                         "determinism is enabled.")
    if alg is None:
      # TODO(b/170668986): more sophisticated algorithm selection
      alg = DEFAULT_ALGORITHM
    alg = random_ops_util.convert_alg_to_int(alg)
    state = non_deterministic_ints(shape=[_get_state_size(alg)],
                                   dtype=SEED_TYPE)
    return cls(state=state, alg=alg)

  @classmethod
  def from_key_counter(cls, key, counter, alg):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says alg may be None to auto-select, but alg is required and is
    # converted directly to an integer id (a null value is not defaulted here).
    # Class method constructor from a scalar key and a counter vector (length state size minus one).
    # Both are converted to int64, shape-checked (assertion error on mismatch), joined with the
    # counter first and the key last into the state, and a new generator is returned.
    """Creates a generator from a key and a counter.

    This constructor only applies if the algorithm is a counter-based algorithm.
    See method `key` for the meaning of "key" and "counter".

    Args:
      key: the key for the RNG, a scalar of type STATE_TYPE.
      counter: a vector of dtype STATE_TYPE representing the initial counter for
        the RNG, whose length is algorithm-specific.,
      alg: the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    counter = _convert_to_state_tensor(counter)
    key = _convert_to_state_tensor(key)
    alg = random_ops_util.convert_alg_to_int(alg)
    counter.shape.assert_is_compatible_with([_get_state_size(alg) - 1])
    key.shape.assert_is_compatible_with([])
    key = array_ops.reshape(key, [1])
    state = array_ops.concat([counter, key], 0)
    return cls(state=state, alg=alg)

  def __init__(self, copy_from=None, state=None, alg=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the argument checks use assertions, so misuse (both or neither source given,
    # copy_from together with alg or state) raises assertion errors, not ValueError.
    # Constructor. Records the current distribution strategy (or none). If copy_from is given, makes
    # a new non-trainable state variable from its state and copies its algorithm. Otherwise both
    # state and alg must be given: alg is converted to an integer id; a variable state is reused
    # directly, any other state is converted to an int64 tensor and wrapped in a new variable.
    # State shape is validated against the algorithm. Creates a variable as a side effect.
    """Creates a generator.

    The new generator will be initialized by one of the following ways, with
    decreasing precedence:
    (1) If `copy_from` is not None, the new generator is initialized by copying
        information from another generator.
    (2) If `state` and `alg` are not None (they must be set together), the new
        generator is initialized by a state.

    Args:
      copy_from: a generator to be copied from.
      state: a vector of dtype STATE_TYPE representing the initial state of the
        RNG, whose length and semantics are algorithm-specific. If it's a
        variable, the generator will reuse it instead of creating a new
        variable.
      alg: the RNG algorithm. Possible values are
        `tf.random.Algorithm.PHILOX` for the Philox algorithm and
        `tf.random.Algorithm.THREEFRY` for the ThreeFry algorithm
        (see paper 'Parallel Random Numbers: As Easy as 1, 2, 3'
        [https://www.thesalmons.org/john/random123/papers/random123sc11.pdf]).
        The string names `"philox"` and `"threefry"` can also be used.
        Note `PHILOX` guarantees the same numbers are produced (given
        the same random state) across all architectures (CPU, GPU, XLA etc).
    """
    # TODO(b/175072242): Remove distribution-strategy dependencies in this file.
    if distribute_lib.has_strategy():
      self._distribution_strategy = distribute_lib.get_strategy()
    else:
      self._distribution_strategy = None
    if copy_from is not None:
      # All other arguments should be None
      assert (alg or state) is None
      self._state_var = self._create_variable(copy_from.state, dtype=STATE_TYPE,
                                              trainable=False)
      self._alg = copy_from.algorithm
    else:
      assert alg is not None and state is not None
      alg = random_ops_util.convert_alg_to_int(alg)
      if isinstance(state, variables.Variable):
        _check_state_shape(state.shape, alg)
        self._state_var = state
      else:
        state = _convert_to_state_tensor(state)
        _check_state_shape(state.shape, alg)
        self._state_var = self._create_variable(state, dtype=STATE_TYPE,
                                                trainable=False)
      self._alg = alg

  def _create_variable(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: creates the generator state variable.
    # Forwards all arguments to the variable constructor, forcing the name to the fixed value used
    # by Keras to find the state, inside a name scope. Returns the variable. Raises ValueError if
    # the result is a sharded variable (state must not be sharded, e.g. by a parameter server
    # partitioner).
    """Creates a variable.

    Args:
      *args: positional arguments passed along to `variables.Variable.
      **kwargs: keyword arguments passed along to `variables.Variable.

    Returns:
      The created variable.
    """
    with ops.name_scope("random_generator"):
      # Make sure we don't change this name since Keras was using this name
      # to filter out the state variable.
      kwargs["name"] = "StateVar"
      v = variables.Variable(*args, **kwargs)
    if isinstance(v, sharded_variable.ShardedVariable):
      # RNG state is an atomic entity representing a 128-bit or
      # 192-bit value, so it mustn't be sharded.
      raise ValueError(
          "tf.random.Generator state is sharded, which is not allowed. When "
          "creating a tf.distribute.experimental.ParameterServerStrategy, "
          "please make sure that the `variable_partitioner` "
          "argument won't shard a "
          "small variable of shape [2] or [3]. Ways to avoid sharding small "
          "variables include setting `variable_partitioner` to None or to "
          "tf.distribute.experimental.partitioners.MinSizePartitioner with a "
          "large enough `min_shard_bytes`.")
    return v

  def reset(self, state):
    # [FUNCTION DOC]
    # Overwrites the generator state with a new one. state is converted to int64 and must have the
    # length required by the current algorithm (assertion error otherwise). Assigns into the state
    # variable (side effect); returns nothing.
    """Resets the generator by a new state.

    See `__init__` for the meaning of "state".

    Args:
      state: the new state.
    """
    state = _convert_to_state_tensor(state)
    state.shape.assert_is_compatible_with([_get_state_size(self.algorithm)])
    self._state_var.assign(state)

  def reset_from_seed(self, seed):
    # [FUNCTION DOC]
    # Re-initializes the state from an integer or vector seed using the generator's own algorithm,
    # assigning to the state variable. Returns nothing.
    """Resets the generator by a new seed.

    See `from_seed` for the meaning of "seed".

    Args:
      seed: the new seed.
    """
    state = create_rng_state(seed, self.algorithm)
    self._state_var.assign(state)

  def reset_from_key_counter(self, key, counter):
    # [FUNCTION DOC]
    # Re-initializes the state from a scalar key and a counter vector (length state size minus one
    # for the current algorithm). Both are converted to int64 and shape-checked (assertion error on
    # mismatch); the state is the counter followed by the key and is assigned into the state
    # variable. Returns nothing.
    """Resets the generator by a new key-counter pair.

    See `from_key_counter` for the meaning of "key" and "counter".

    Args:
      key: the new key.
      counter: the new counter.
    """
    counter = _convert_to_state_tensor(counter)
    key = _convert_to_state_tensor(key)
    counter.shape.assert_is_compatible_with(
        [_get_state_size(self.algorithm) - 1])
    key.shape.assert_is_compatible_with([])
    key = array_ops.reshape(key, [1])
    state = array_ops.concat([counter, key], 0)
    self._state_var.assign(state)

  @property
  def state(self):
    # [FUNCTION DOC]
    # Read-only property returning the internal state variable (int64 vector whose length depends
    # on the algorithm). The variable itself, not a copy, is returned.
    """The internal state of the RNG."""
    return self._state_var

  @property
  def algorithm(self):
    # [FUNCTION DOC]
    # Read-only property returning the algorithm id stored at construction (an integer, or a scalar
    # integer tensor if one was supplied through a variable state).
    """The RNG algorithm id (a Python integer or scalar integer Tensor)."""
    return self._alg

  def _standard_normal(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: core sampler for normal.
    # Advances the state by the number of elements, then calls the stateless normal op with the
    # derived key and counter. shape is an integer tensor, dtype the float type. Returns a tensor
    # of standard normal samples. Side effect: state variable is updated.
    key, counter = self._prepare_key_counter(shape)
    return gen_stateless_random_ops_v2.stateless_random_normal_v2(
        shape, key=key, counter=counter, dtype=dtype, alg=self.algorithm)

  @property
  def key(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises ValueError for non counter-based algorithms, but the
    # check only tests membership in the Algorithm enum values, so any listed algorithm passes
    # (including auto-select) and only unknown ids raise.
    # KNOWN ISSUE: the error path calls an error-message function missing from the stateless
    # random ops module, so an unknown id actually raises an attribute error.
    # Read-only property returning the last element of the state (the key) as a scalar tensor.
    # Raises ValueError with the unsupported-algorithm message for unknown algorithm ids.
    """The 'key' part of the state of a counter-based RNG.

    For a counter-base RNG algorithm such as Philox and ThreeFry (as
    described in paper 'Parallel Random Numbers: As Easy as 1, 2, 3'
    [https://www.thesalmons.org/john/random123/papers/random123sc11.pdf]),
    the RNG state consists of two parts: counter and key. The output is
    generated via the formula: output=hash(key, counter), i.e. a hashing of
    the counter parametrized by the key. Two RNGs with two different keys can
    be thought as generating two independent random-number streams (a stream
    is formed by increasing the counter).

    Returns:
      A scalar which is the 'key' part of the state, if the RNG algorithm is
        counter-based; otherwise it raises a ValueError.
    """
    alg = self.algorithm
    if alg in (a.value for a in random_ops_util.Algorithm):
      return self._state_var[-1]
    else:
      raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))

  def _skip_single_var(self, var, delta):
    # [FUNCTION DOC]
    # HELPER: per-variable worker for skip.
    # Marks the variable as accessed, then runs the read-and-skip op on its handle with the
    # algorithm cast to int32 and delta cast to uint64. Returns the int64 tensor holding the state
    # (counter and key) as it was before advancing. Side effect: updates the variable in place.
    resource_variable_ops.variable_accessed(var)
    # TODO(wangpeng): Cache the cast algorithm instead of casting everytime.
    return gen_stateful_random_ops.rng_read_and_skip(
        var.handle,
        alg=math_ops.cast(self.algorithm, dtypes.int32),
        delta=math_ops.cast(delta, dtypes.uint64))

  def skip(self, delta):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring returns only "a Tensor of int64" without saying it is the
    # pre-advance counter-plus-key state, which other methods rely on.
    # Advances the counter of the state by delta (integer, scalar) and returns the old state
    # (counter and key) as an int64 tensor. When saving a non-distributed model, applies the
    # update directly. Under a strategy, in cross-replica context (or CentralStorage) the update is
    # routed through the strategy's extended update and the function is marked unsaveable in
    # cross-replica context; otherwise the update runs directly. Mutates the state variable.
    # Contains a nested function that applies the skip to one variable.
    """Advance the counter of a counter-based RNG.

    Args:
      delta: the amount of advancement. The state of the RNG after
        `skip(n)` will be the same as that after `normal([n])`
        (or any other distribution). The actual increment added to the
        counter is an unspecified implementation detail.

    Returns:
      A `Tensor` of type `int64`.
    """

    def update_fn(v):
      # [FUNCTION DOC]
      # HELPER: update callback for skip. Applies the skip by delta to variable v and returns the
      # old state tensor.
      return self._skip_single_var(v, delta)
    # TODO(b/170515001): Always call strategy.extended.update after calling it
    #   from both replica context and cross-replica context is supported.
    if values_util.is_saving_non_distributed():
      # Assumes replica context with replica_id=0, since we only save the first
      # replica.
      return update_fn(self.state)
    if self._distribution_strategy is not None:
      with distribute_lib.enter_or_assert_strategy(self._distribution_strategy):
        if distribute_lib.in_cross_replica_context():
          # Code that operates on all replicas of a variable cannot be saved
          # without retracing.
          values_util.mark_as_unsaveable()
        if (distribute_lib.in_cross_replica_context() or
            "CentralStorage" in type(self._distribution_strategy).__name__):
          # In cross-replica context we need to use strategy.extended.update.
          # In CentralStorageStrategy we also need to use
          # strategy.extended.update (even for replica context),
          # because variable updates here must be within merge_call.
          return distribute_lib.get_strategy().extended.update(
              self.state, update_fn)
    return update_fn(self.state)

  def _preprocess_key(self, key):
    # [FUNCTION DOC]
    # HELPER: per-replica key derivation.
    # Without a distribution strategy returns the key unchanged. Otherwise, inside the strategy
    # scope, if there is a replica id, hashes the key with the replica id (as a two-element uint64
    # counter) via the stateless full-integer op so each replica gets an independent stream.
    # Returns the key tensor (uint64).
    if self._distribution_strategy is None:
      return key
    with distribute_lib.enter_or_assert_strategy(self._distribution_strategy):
      replica_id = get_replica_id()
      if replica_id is not None:
        replica_id = array_ops_stack.stack([replica_id, 0], axis=0)
        replica_id = math_ops.cast(replica_id, dtypes.uint64)
        # Conceptually: key = hash(key, replica_id)
        key = gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(
            shape=[1], key=key, counter=replica_id, dtype=dtypes.uint64,
            alg=self.algorithm)
      return key

  def _prepare_key_counter(self, shape):
    # [FUNCTION DOC]
    # HELPER: common preamble for all sampling methods.
    # Advances the state by the product of shape (the number of samples), splits the returned old
    # state into counter and key (reinterpreted as uint64), and applies per-replica key derivation.
    # Returns a pair: key, counter. Side effect: the state variable is advanced.
    delta = math_ops.reduce_prod(shape)
    counter_key = self.skip(delta)
    counter_size = _get_counter_size(self.algorithm)
    counter = array_ops.bitcast(counter_key[:counter_size], dtypes.uint64)
    key = array_ops.bitcast(counter_key[counter_size:counter_size + 1],
                            dtypes.uint64)
    key = self._preprocess_key(key)
    return key, counter

  # The following functions return a tensor and as a side effect update
  # self._state_var.
  def normal(self, shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
             name=None):
    # [FUNCTION DOC]
    # Samples a normal distribution. shape: 1-D integer tensor or list; mean and stddev: scalar
    # values or tensors (defaults 0 and 1) converted to dtype; dtype: float output type (default
    # float32); name: optional scope name. Draws standard normal values, multiplies by stddev and
    # adds mean. Returns a tensor of the given shape. Advances the generator state.
    """Outputs random values from a normal distribution.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
        distribution.
      stddev: A 0-D Tensor or Python value of type `dtype`. The standard
        deviation of the normal distribution.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random normal values.
    """
    with ops.name_scope(name, "stateful_normal", [shape, mean, stddev]) as name:
      shape = _shape_tensor(shape)
      mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
      stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
      rnd = self._standard_normal(shape, dtype=dtype)
      return math_ops.add(rnd * stddev, mean, name=name)

  def _truncated_normal(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: core sampler for truncated_normal. Advances the state, then calls the stateless
    # truncated normal op (standard normal truncated at two standard deviations) for the given
    # shape and dtype. Returns the sample tensor.
    key, counter = self._prepare_key_counter(shape)
    return gen_stateless_random_ops_v2.stateless_truncated_normal_v2(
        shape=shape, key=key, counter=counter, dtype=dtype, alg=self.algorithm)

  def truncated_normal(self, shape,
                       mean=0.0,
                       stddev=1.0,
                       dtype=dtypes.float32,
                       name=None):
    # [FUNCTION DOC]
    # Samples a truncated normal distribution. shape: 1-D integer tensor or list; mean, stddev:
    # scalars or tensors (defaults 0 and 1); dtype: float type (default float32); name optional.
    # Draws truncated standard normals (resampled beyond two standard deviations), scales by
    # stddev and shifts by mean. Returns a tensor of the given shape. Advances the generator state.
    """Outputs random values from a truncated normal distribution.


    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than
    2 standard deviations from the mean are dropped and re-picked.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
        truncated normal distribution.
      stddev: A 0-D Tensor or Python value of type `dtype`. The standard
        deviation of the normal distribution, before truncation.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random truncated normal
        values.
    """
    with ops.name_scope(
        name, "truncated_normal", [shape, mean, stddev]) as name:
      shape_tensor = _shape_tensor(shape)
      mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
      stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
      rnd = self._truncated_normal(shape_tensor, dtype=dtype)
      mul = rnd * stddev_tensor
      return math_ops.add(mul, mean_tensor, name=name)

  def _uniform(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: core sampler for floating point uniform draws. Advances the state and calls the
    # stateless uniform op, producing values in the unit interval of the given float dtype for
    # the given shape tensor. Returns the sample tensor.
    key, counter = self._prepare_key_counter(shape)
    return gen_stateless_random_ops_v2.stateless_random_uniform_v2(
        shape=shape, key=key, counter=counter, dtype=dtype, alg=self.algorithm)

  def _uniform_full_int(self, shape, dtype, name=None):
    # [FUNCTION DOC]
    # HELPER: core sampler for full-range integer draws (used by uniform, uniform_full_int and key
    # generation). Advances the state and calls the stateless full-integer op for the shape tensor
    # and integer dtype, with an optional op name. Returns the integer tensor.
    key, counter = self._prepare_key_counter(shape)
    return gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(
        shape=shape,
        key=key,
        counter=counter,
        dtype=dtype,
        alg=self.algorithm,
        name=name)

  def uniform(self, shape, minval=0, maxval=None,
              dtype=dtypes.float32, name=None):
    # [FUNCTION DOC]
    # Samples a uniform distribution. shape: 1-D integer tensor or list; minval (default 0) and
    # maxval (default none); dtype (default float32); name optional. For integer dtypes minval and
    # maxval must be both given or both none (none means the full integer range), otherwise a
    # ValueError is raised. For floats a missing maxval becomes 1. Integer ranges use the integer
    # uniform op; floats scale a unit uniform sample by the range and add minval. Returns a tensor
    # of the given shape. Advances the generator state.
    """Outputs random values from a uniform distribution.


    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded. (For float numbers especially
    low-precision types like bfloat16, because of
    rounding, the result may sometimes include `maxval`.)

    For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `maxval - minval` is an exact power of two.  The bias is small for values of
    `maxval - minval` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    For full-range random integers, pass `minval=None` and `maxval=None` with an
    integer `dtype` (for integer dtypes, `minval` and `maxval` must be both
    `None` or both not `None`).

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      minval: A Tensor or Python value of type `dtype`, broadcastable with
        `shape` (for integer types, broadcasting is not supported, so it needs
        to be a scalar). The lower bound (included) on the range of random
        values to generate. Pass `None` for full-range integers. Defaults to 0.
      maxval: A Tensor or Python value of type `dtype`, broadcastable with
        `shape` (for integer types, broadcasting is not supported, so it needs
        to be a scalar). The upper bound (excluded) on the range of random
        values to generate. Pass `None` for full-range integers. Defaults to 1
        if `dtype` is floating point.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random uniform values.

    Raises:
      ValueError: If `dtype` is integral and `maxval` is not specified.
    """
    dtype = dtypes.as_dtype(dtype)
    if dtype.is_integer:
      if (minval is None) != (maxval is None):
        raise ValueError("For integer dtype {}, minval and maxval must be both "
                         "`None` or both non-`None`; got minval={} and "
                         "maxval={}".format(dtype, minval, maxval))
    elif maxval is None:
      maxval = 1
    with ops.name_scope(name, "stateful_uniform",
                        [shape, minval, maxval]) as name:
      shape = _shape_tensor(shape)
      if dtype.is_integer and minval is None:
        return self._uniform_full_int(shape=shape, dtype=dtype, name=name)
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
      if dtype.is_integer:
        key, counter = self._prepare_key_counter(shape)
        return gen_stateless_random_ops_v2.stateless_random_uniform_int_v2(
            shape=shape,
            key=key,
            counter=counter,
            minval=minval,
            maxval=maxval,
            alg=self.algorithm,
            name=name)
      else:
        rnd = self._uniform(shape=shape, dtype=dtype)
        return math_ops.add(rnd * (maxval - minval), minval, name=name)

  def uniform_full_int(self, shape, dtype=dtypes.uint64, name=None):
    # [FUNCTION DOC]
    # Draws integers uniformly over the whole range of an integer type. shape: output shape; dtype:
    # integer type (default uint64); name optional. Returns a tensor of that shape and type.
    # Advances the generator state.
    """Uniform distribution on an integer type's entire range.

    This method is the same as setting `minval` and `maxval` to `None` in the
    `uniform` method.

    Args:
      shape: the shape of the output.
      dtype: (optional) the integer type, default to uint64.
      name: (optional) the name of the node.

    Returns:
      A tensor of random numbers of the required shape.
    """
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope(name, "stateful_uniform_full_int",
                        [shape]) as name:
      shape = _shape_tensor(shape)
      return self._uniform_full_int(shape=shape, dtype=dtype, name=name)

  def binomial(self, shape, counts, probs, dtype=dtypes.int32, name=None):
    # [FUNCTION DOC]
    # Samples a binomial distribution. shape: output shape; counts and probs: tensors of trial
    # counts and success probabilities, broadcastable with each other and with the trailing
    # dimensions of shape; dtype: output type (default int32); name optional. Calls the stateful
    # binomial op directly on the state variable handle, so the state is updated by the op.
    # Returns a tensor of the given shape.
    """Outputs random values from a binomial distribution.

    The generated values follow a binomial distribution with specified count and
    probability of success parameters.

    Example:

    ```python
    counts = [10., 20.]
    # Probability of success.
    probs = [0.8]

    rng = tf.random.Generator.from_seed(seed=234)
    binomial_samples = rng.binomial(shape=[2], counts=counts, probs=probs)


    counts = ... # Shape [3, 1, 2]
    probs = ...  # Shape [1, 4, 2]
    shape = [3, 4, 3, 4, 2]
    rng = tf.random.Generator.from_seed(seed=1717)
    # Sample shape will be [3, 4, 3, 4, 2]
    binomial_samples = rng.binomial(shape=shape, counts=counts, probs=probs)
    ```


    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      counts: Tensor. The counts of the binomial distribution. Must be
        broadcastable with `probs`, and broadcastable with the rightmost
        dimensions of `shape`.
      probs: Tensor. The probability of success for the
        binomial distribution. Must be broadcastable with `counts` and
        broadcastable with the rightmost dimensions of `shape`.
      dtype: The type of the output. Default: tf.int32
      name: A name for the operation (optional).

    Returns:
      samples: A Tensor of the specified shape filled with random binomial
        values.  For each i, each samples[i, ...] is an independent draw from
        the binomial distribution on counts[i] trials with probability of
        success probs[i].
    """
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope(name, "binomial", [shape, counts, probs]) as name:
      counts = ops.convert_to_tensor(counts, name="counts")
      probs = ops.convert_to_tensor(probs, name="probs")
      shape_tensor = _shape_tensor(shape)
      return gen_stateful_random_ops.stateful_random_binomial(
          self.state.handle,
          self.algorithm,
          shape=shape_tensor,
          counts=counts,
          probs=probs,
          dtype=dtype,
          name=name)

  # TODO(wangpeng): implement other distributions

  def _make_int64_keys(self, shape=()):
    # [FUNCTION DOC]
    # HELPER: key source for make_seeds and split. Draws independent full-range int64 values of
    # the given shape (default scalar) by calling full-range uniform sampling, which advances the
    # state. Returns an int64 tensor.
    # New independent keys are generated via
    # `new_key[i] = hash(old_key, counter+i)`, which is exactly what
    # `uniform_full_int(dtype=int64)` does for PhiloxRandom_64_128_128 and
    # ThreeFry_64_64_64.
    return self.uniform_full_int(shape=shape, dtype=dtypes.int64)

  def make_seeds(self, count=1):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the unknown-algorithm path calls an error-message function missing from the
    # stateless random ops module, so it actually raises an attribute error.
    # Generates seed pairs for stateless random ops. count: number of pairs (default 1). Draws
    # count int64 keys, pairs each with zero and stacks them. Returns an int64 tensor of shape
    # two by count. Intended to raise ValueError for unknown algorithm ids. Advances the
    # generator state.
    """Generates seeds for stateless random ops.

    For example:

    ```python
    seeds = get_global_generator().make_seeds(count=10)
    for i in range(10):
      seed = seeds[:, i]
      numbers = stateless_random_normal(shape=[2, 3], seed=seed)
      ...
    ```

    Args:
      count: the number of seed pairs (note that stateless random ops need a
        pair of seeds to invoke).

    Returns:
      A tensor of shape [2, count] and dtype int64.
    """
    alg = self.algorithm
    if alg in (a.value for a in random_ops_util.Algorithm):
      keys = self._make_int64_keys(shape=[count])
      # The two seeds for stateless random ops don't have individual semantics
      # and are scrambled together, so setting one to zero is fine.
      zeros = array_ops.zeros_like(keys)
      return array_ops_stack.stack([keys, zeros])
    else:
      raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))

  def split(self, count=1):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the unknown-algorithm path calls an error-message function missing from the
    # stateless random ops module, so it actually raises an attribute error.
    # Creates independent child generators. count: how many (default 1). Draws count int64 keys
    # (advancing this generator), and for each builds a new generator with zero counter and that
    # key as state, using the same algorithm. Returns a list of count generators, placed on the
    # current device. Intended to raise ValueError for unknown algorithm ids.
    """Returns a list of independent `Generator` objects.

    Two generators are independent of each other in the sense that the
    random-number streams they generate don't have statistically detectable
    correlations. The new generators are also independent of the old one.
    The old generator's state will be changed (like other random-number
    generating methods), so two calls of `split` will return different
    new generators.

    For example:

    ```python
    gens = get_global_generator().split(count=10)
    for gen in gens:
      numbers = gen.normal(shape=[2, 3])
      # ...
    gens2 = get_global_generator().split(count=10)
    # gens2 will be different from gens
    ```

    The new generators will be put on the current device (possible different
    from the old generator's), for example:

    ```python
    with tf.device("/device:CPU:0"):
      gen = Generator(seed=1234)  # gen is on CPU
    with tf.device("/device:GPU:0"):
      gens = gen.split(count=10)  # gens are on GPU
    ```

    Args:
      count: the number of generators to return.

    Returns:
      A list (length `count`) of `Generator` objects independent of each other.
      The new generators have the same RNG algorithm as the old one.
    """
    def _key_to_state(alg, key):
      # [FUNCTION DOC]
      # HELPER: state builder for split. Returns a list of zeros (one fewer than the state size
      # for alg) followed by the key, i.e. a zero counter and the given key.
      # Padding with zeros on the left. The zeros will be the counter.
      return [0] * (_get_state_size(alg) - 1) + [key]

    alg = self.algorithm
    if alg in (a.value for a in random_ops_util.Algorithm):
      keys = self._make_int64_keys(shape=[count])
      return [Generator(state=_key_to_state(alg, key), alg=alg)
              for key in array_ops_stack.unstack(keys, num=count)]
    else:
      raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))


# It's not safe to create TF ops before `init_google` is called, so this is
# initialized to None and get a value the first time `get_global_generator` is
# called.
global_generator = None


@tf_export("random.get_global_generator",
           "random.experimental.get_global_generator")
def get_global_generator():
  # [FUNCTION DOC]
  # Returns the process-wide generator, creating it lazily on first use from non-deterministic
  # state (built in an init scope so it is created outside any function graph) on the default
  # device. Raises RuntimeError if the global generator is not yet set and op determinism is
  # enabled. Side effect: sets the module-level global generator.
  """Retrieves the global generator.

  This function will create the global generator the first time it is called,
  and the generator will be placed at the default device at that time, so one
  needs to be careful when this function is first called. Using a generator
  placed on a less-ideal device will incur performance regression.

  Returns:
    The global `tf.random.Generator` object.
  """
  global global_generator
  if global_generator is None:
    if config.is_op_determinism_enabled():
      raise RuntimeError('"get_global_generator" cannot be called if '  # pylint: disable=g-doc-exception
                         "determinism is enabled, unless "
                         '"set_global_generator" has already been called. '
                         'Please call "set_global_generator" first.')
    with ops.init_scope():
      global_generator = Generator.from_non_deterministic_state()
  return global_generator


@tf_export("random.set_global_generator",
           "random.experimental.set_global_generator")
def set_global_generator(generator):
  # [FUNCTION DOC]
  # Replaces the module-level global generator with the given Generator object (no validation of
  # its type). Returns nothing. Previously compiled functions that captured the old generator are
  # not affected.
  """Replaces the global generator with another `Generator` object.

  This function replaces the global generator with the provided `generator`
  object.
  A random number generator utilizes a `tf.Variable` object to store its state.
  The user shall be aware of caveats how `set_global_generator` interacts with
  `tf.function`:

  - tf.function puts restrictions on Variable creation thus one cannot freely
    create a new random generator instance inside `tf.function`.
    To call `set_global_generator` inside `tf.function`, the generator instance
    must have already been created eagerly.
  - tf.function captures the Variable during trace-compilation, thus a compiled
    f.function will not be affected `set_global_generator` as demonstrated by
    random_test.py/RandomTest.testResetGlobalGeneratorBadWithDefun .

  For most use cases, avoid calling `set_global_generator` after program
  initialization, and prefer to reset the state of the existing global generator
  instead, such as,

  >>> rng = tf.random.get_global_generator()
  >>> rng.reset_from_seed(30)


  Args:
    generator: the new `Generator` object.
  """
  global global_generator
  global_generator = generator
