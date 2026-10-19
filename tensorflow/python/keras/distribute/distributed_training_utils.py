# [MODULE DOC]
# HELPER: helper module used by Keras training code that runs under a distribution strategy.
# Small utilities for distributed training: a check for global batch size support, a wrapper
# that calls functions touching replica-local variables in the correct (per-replica) context,
# and a predicate identifying distributed variables. Exposes three plain functions and has no
# import-time side effects beyond importing distribute, values, backend and variables modules.
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
"""Utilities related to distributed training."""
# pylint:disable=protected-access

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables


# TODO(b/118776054): Currently we support global batch size for TPUStrategy and
# core MirroredStrategy only. Remove this check when contrib MirroredStrategy is
# no longer needed.
def global_batch_size_supported(distribution_strategy):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the name suggests a boolean predicate, but the function returns whatever the
  # strategy's extended object stores in its private global-batch-size attribute, without
  # converting it to a boolean.
  # Reports whether the given distribution strategy interprets batch sizes as global batch
  # sizes. Input: distribution_strategy, a tf.distribute strategy object. Output: the value of
  # the strategy extended object's private global batch size attribute (normally true or false).
  # Raises an attribute error if the strategy lacks that attribute. No side effects.
  return distribution_strategy.extended._global_batch_size  # pylint: disable=protected-access


def call_replica_local_fn(fn, *args, **kwargs):
  # [FUNCTION DOC]
  # Calls fn with the given positional and keyword arguments, making sure that code touching
  # replica-local variables runs per replica. An optional keyword argument named strategy is
  # removed from the keyword arguments and used as the strategy; otherwise the current strategy
  # is used if one exists. If the strategy is not a TPU strategy and the caller is currently in
  # a cross-replica context, fn is run once per replica inside the strategy scope and the
  # per-replica result is returned. In all other cases fn is called directly.
  # Returns whatever fn (or the per-replica call) returns. Exceptions from fn propagate.
  # Side effect: enters the strategy scope while calling; the strategy keyword is consumed.
  """Call a function that uses replica-local variables.

  This function correctly handles calling `fn` in a cross-replica
  context.

  Args:
    fn: The function to call.
    *args: Positional arguments to the `fn`.
    **kwargs: Keyword argument to `fn`.

  Returns:
    The result of calling `fn`.
  """
  # TODO(b/132666209): Remove this function when we support assign_*
  # for replica-local variables.
  strategy = None
  if 'strategy' in kwargs:
    strategy = kwargs.pop('strategy')
  else:
    if distribute_lib.has_strategy():
      strategy = distribute_lib.get_strategy()

  # TODO(b/120571621): TPUStrategy does not implement replica-local variables.
  is_tpu = backend.is_tpu_strategy(strategy)
  if ((not is_tpu) and strategy and distribute_lib.in_cross_replica_context()):
    with strategy.scope():
      return strategy.extended.call_for_each_replica(fn, args, kwargs)
  return fn(*args, **kwargs)


def is_distributed_variable(v):
  # [FUNCTION DOC]
  # Predicate: returns true only if v is both a distributed-values container and a variable
  # object. Input: any object v. Output: boolean. No exceptions or side effects.
  """Returns whether `v` is a distributed variable."""
  return (isinstance(v, values_lib.DistributedValues) and
          isinstance(v, variables.Variable))
