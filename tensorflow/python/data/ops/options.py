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
# [MODULE DOC]
# Public API for specifying tf.data options. Defines enums for autotune algorithm, auto-shard
# policy and external state policy, grouped option classes (autotune, distribute, optimization,
# service, threading) and the top-level Options class that aggregates them. Each option class
# declares its settable options with types and docs, and converts to and from the dataset options
# protocol buffer messages. Import-time side effects: exports public names through the API export
# decorators and registers deprecated endpoints.
"""API for specifying `tf.data` options."""

import enum
import platform

from absl import logging

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import model_pb2
from tensorflow.python.data.ops import test_mode
from tensorflow.python.data.util import options as options_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.AutotuneAlgorithm")
class AutotuneAlgorithm(enum.Enum):
  # [CLASS DOC]
  # Enumeration of the autotuning algorithms (default, hill climb, gradient descent, max
  # parallelism, stage based) with conversion to and from the matching model protocol buffer
  # enum values.
  """Represents the type of autotuning algorithm to use.

  DEFAULT: The default behavior is implementation specific and may change over
  time.

  HILL_CLIMB: In each optimization step, this algorithm chooses the optimal
  parameter and increases its value by 1.

  GRADIENT_DESCENT: In each optimization step, this algorithm updates the
  parameter values in the optimal direction.

  MAX_PARALLELISM: Similar to HILL_CLIMB but uses a relaxed stopping condition,
  allowing the optimization to oversubscribe the CPU.

  STAGE_BASED: In each optimization step, this algorithm chooses the worst
  bottleneck parameter and increases its value by 1.
  """
  DEFAULT = 0
  HILL_CLIMB = 1
  GRADIENT_DESCENT = 2
  MAX_PARALLELISM = 3
  STAGE_BASED = 4

  @classmethod
  def _to_proto(cls, obj):
    # [FUNCTION DOC]
    # HELPER: serialization support for the autotune options.
    # KNOWN ISSUE: the error message for an unsupported value lists default, hill climb,
    # gradient descent and stage based but omits max parallelism, although that value is
    # supported.
    # Class method. Input: an algorithm member. Returns the matching protocol buffer enum value;
    # raises a value error if the member matches none.
    if obj == cls.DEFAULT:
      return model_pb2.AutotuneAlgorithm.DEFAULT
    if obj == cls.HILL_CLIMB:
      return model_pb2.AutotuneAlgorithm.HILL_CLIMB
    if obj == cls.GRADIENT_DESCENT:
      return model_pb2.AutotuneAlgorithm.GRADIENT_DESCENT
    if obj == cls.MAX_PARALLELISM:
      return model_pb2.AutotuneAlgorithm.MAX_PARALLELISM
    if obj == cls.STAGE_BASED:
      return model_pb2.AutotuneAlgorithm.STAGE_BASED
    raise ValueError(
        f"Invalid `obj.` Supported values include `DEFAULT`, `HILL_CLIMB` "
        f"`GRADIENT_DESCENT`, and `STAGE_BASED`. Got {obj.name}.")

  @classmethod
  def _from_proto(cls, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support for the autotune options.
    # KNOWN ISSUE: the error message omits max parallelism from the list of supported values.
    # Class method. Input: protocol buffer enum value. Returns the matching algorithm member;
    # raises a value error for an unknown value.
    if pb == model_pb2.AutotuneAlgorithm.DEFAULT:
      return cls.DEFAULT
    if pb == model_pb2.AutotuneAlgorithm.HILL_CLIMB:
      return cls.HILL_CLIMB
    if pb == model_pb2.AutotuneAlgorithm.GRADIENT_DESCENT:
      return cls.GRADIENT_DESCENT
    if pb == model_pb2.AutotuneAlgorithm.MAX_PARALLELISM:
      return cls.MAX_PARALLELISM
    if pb == model_pb2.AutotuneAlgorithm.STAGE_BASED:
      return cls.STAGE_BASED
    raise ValueError(
        f"Invalid `pb.` Supported values include `DEFAULT`, `HILL_CLIMB`, "
        f"`GRADIENT_DESCENT` and `STAGE_BASED`. Got {pb}.")


@tf_export("data.experimental.AutoShardPolicy")
class AutoShardPolicy(enum.IntEnum):
  # [CLASS DOC]
  # Integer enumeration of auto-sharding policies (off, auto, file, data, hint) with conversion
  # to and from the dataset options protocol buffer enum. Values must stay in sync with the
  # data service sharding policy (see the lint marker in the code).
  """Represents the type of auto-sharding to use.

  OFF: No sharding will be performed.

  AUTO: Attempts FILE-based sharding, falling back to DATA-based sharding.

  FILE: Shards by input files (i.e. each worker will get a set of files to
  process). When this option is selected, make sure that there is at least as
  many files as workers. If there are fewer input files than workers, a runtime
  error will be raised.

  DATA: Shards by elements produced by the dataset. Each worker will process the
  whole dataset and discard the portion that is not for itself. Note that for
  this mode to correctly partitions the dataset elements, the dataset needs to
  produce elements in a deterministic order.

  HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
  placeholder to replace with `shard(num_workers, worker_index)`.
  """

  # LINT.IfChange
  OFF = -1
  AUTO = 0
  FILE = 1
  DATA = 2
  HINT = 3
  # LINT.ThenChange(//tensorflow/python/data/experimental/ops/data_service_ops.py:tf_data_service_sharding_policy)

  @classmethod
  def _to_proto(cls, obj):
    # [FUNCTION DOC]
    # HELPER: serialization support for the distribute options.
    # Class method. Input: a policy member. Returns the matching protocol buffer enum value;
    # raises a value error if there is no match.
    """Convert enum to proto."""
    if obj == cls.OFF:
      return dataset_options_pb2.AutoShardPolicy.OFF
    if obj == cls.FILE:
      return dataset_options_pb2.AutoShardPolicy.FILE
    if obj == cls.DATA:
      return dataset_options_pb2.AutoShardPolicy.DATA
    if obj == cls.AUTO:
      return dataset_options_pb2.AutoShardPolicy.AUTO
    if obj == cls.HINT:
      return dataset_options_pb2.AutoShardPolicy.HINT
    raise ValueError(
        f"Invalid `obj.` Supported values include `OFF`, `FILE`, `DATA`,"
        f"`AUTO`, and `HINT`. Got {obj.name}."
    )

  @classmethod
  def _from_proto(cls, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support for the distribute options.
    # Class method. Input: protocol buffer enum value. Returns the matching policy member;
    # raises a value error for an unknown value.
    """Convert proto to enum."""
    if pb == dataset_options_pb2.AutoShardPolicy.OFF:
      return cls.OFF
    if pb == dataset_options_pb2.AutoShardPolicy.FILE:
      return cls.FILE
    if pb == dataset_options_pb2.AutoShardPolicy.DATA:
      return cls.DATA
    if pb == dataset_options_pb2.AutoShardPolicy.AUTO:
      return cls.AUTO
    if pb == dataset_options_pb2.AutoShardPolicy.HINT:
      return cls.HINT
    raise ValueError(
        f"Invalid `pb.` Supported values include `OFF`, `FILE`, `DATA`,"
        f"`AUTO`, and `HINT`. Got {pb}."
    )


@tf_export("data.experimental.ExternalStatePolicy")
class ExternalStatePolicy(enum.Enum):
  # [CLASS DOC]
  # Enumeration of how external state is handled on serialization (warn, ignore, fail), with
  # conversion to and from the dataset options protocol buffer policy enum.
  """Represents how to handle external state during serialization.

  See the `tf.data.Options.experimental_external_state_policy` documentation
  for more information.
  """
  WARN = 0
  IGNORE = 1
  FAIL = 2

  @classmethod
  def _to_proto(cls, obj):
    # [FUNCTION DOC]
    # HELPER: serialization support for the top-level options.
    # Class method. Input: a policy member. Returns the matching protocol buffer policy value;
    # raises a value error if there is no match.
    """Convert enum to proto."""
    if obj == cls.IGNORE:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_IGNORE
    if obj == cls.FAIL:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_FAIL
    if obj == cls.WARN:
      return dataset_options_pb2.ExternalStatePolicy.POLICY_WARN
    raise ValueError(
        f"Invalid `obj.` Supported values include `POLICY_IGNORE`,"
        f"`POLICY_FAIL`, `POLICY_WARN`. Got {obj.name}.")

  @classmethod
  def _from_proto(cls, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support for the top-level options.
    # Class method. Input: protocol buffer policy value. Returns the matching member; raises a
    # value error for an unknown value.
    """Convert proto to enum."""
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_IGNORE:
      return cls.IGNORE
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_FAIL:
      return cls.FAIL
    if pb == dataset_options_pb2.ExternalStatePolicy.POLICY_WARN:
      return cls.WARN
    raise ValueError(
        f"Invalid `pb.` Supported values include `POLICY_IGNORE`,"
        f"`POLICY_FAIL`, `POLICY_WARN`. Got {pb}.")


@tf_export("data.experimental.AutotuneOptions")
class AutotuneOptions(options_lib.OptionsBase):
  # [CLASS DOC]
  # Options controlling autotuning of dataset performance: enabled flag, CPU budget, RAM budget,
  # algorithm, initial parallelism and minimum parallelism. All default to unset (nothing) and
  # are declared with typed option descriptors. Converts to and from the autotune options
  # protocol buffer message.
  """Represents options for autotuning dataset performance.

  ```python
  options = tf.data.Options()
  options.autotune.enabled = False
  dataset = dataset.with_options(options)
  ```
  """

  enabled = options_lib.create_option(
      name="enabled",
      ty=bool,
      docstring="Whether to automatically tune performance knobs. If None, "
      "defaults to True.")

  cpu_budget = options_lib.create_option(
      name="cpu_budget",
      ty=int,
      docstring="When autotuning is enabled (through `autotune`), determines "
      "the CPU budget to use. Values greater than the number of schedulable "
      "CPU cores are allowed but may result in CPU contention. If None, "
      "defaults to the number of schedulable CPU cores.")

  ram_budget = options_lib.create_option(
      name="ram_budget",
      ty=int,
      docstring="When autotuning is enabled (through `autotune`), determines "
      "the RAM budget to use. Values greater than the available RAM in bytes "
      "may result in OOM. If None, defaults to half of the available RAM in "
      "bytes.")

  autotune_algorithm = options_lib.create_option(
      name="autotune_algorithm",
      ty=AutotuneAlgorithm,
      docstring="When autotuning is enabled (through `autotune`), determines "
      "the algorithm to use.")

  initial_parallelism = options_lib.create_option(
      name="initial_parallelism",
      ty=int,
      docstring=(
          "The initial parallelism to use for parallel transformations before"
          " autotune has a chance to run. A higher value can help with quick"
          " startup, but may cause the ram_budget to temporarily be exceeded."
          " Memory-sensitive datasets should consider setting this to `1` to"
          " avoid running out of memory. Defaults to 16."
      ),
  )

  min_parallelism = options_lib.create_option(
      name="min_parallelism",
      ty=int,
      docstring=(
          "When true, `.map(num_parallel_calls=AUTOTUNE)` and"
          " `.batch(num_parallel_calls=AUTOTUNE)` will be at least"
          " parallelized by `min_parallelism` threads."
      ),
  )

  def _to_proto(self):
    # [FUNCTION DOC]
    # HELPER: serialization for with-options and dataset serialization.
    # Returns an autotune options message with every option that is set copied in (the algorithm
    # is converted to its protocol buffer enum); unset options are left out.
    pb = dataset_options_pb2.AutotuneOptions()
    if self.enabled is not None:
      pb.enabled = self.enabled
    if self.cpu_budget is not None:
      pb.cpu_budget = self.cpu_budget
    if self.ram_budget is not None:
      pb.ram_budget = self.ram_budget
    if self.autotune_algorithm is not None:
      pb.autotune_algorithm = AutotuneAlgorithm._to_proto(  # pylint: disable=protected-access
          self.autotune_algorithm)
    if self.initial_parallelism is not None:
      pb.initial_parallelism = self.initial_parallelism
    if self.min_parallelism is not None:
      pb.min_parallelism = self.min_parallelism
    return pb

  def _from_proto(self, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization counterpart of the to-message method.
    # Input: an autotune options message. Sets on this object each option whose optional field is
    # present in the message (converting the algorithm enum); other options are untouched.
    # Returns nothing.
    if pb.WhichOneof("optional_enabled") is not None:
      self.enabled = pb.enabled
    if pb.WhichOneof("optional_cpu_budget") is not None:
      self.cpu_budget = pb.cpu_budget
    if pb.WhichOneof("optional_ram_budget") is not None:
      self.ram_budget = pb.ram_budget
    if pb.WhichOneof("optional_autotune_algorithm") is not None:
      self.autotune_algorithm = AutotuneAlgorithm._from_proto(  # pylint: disable=protected-access
          pb.autotune_algorithm)
    if pb.WhichOneof("optional_initial_parallelism") is not None:
      self.initial_parallelism = pb.initial_parallelism
    if pb.WhichOneof("optional_min_parallelism") is not None:
      self.min_parallelism = pb.min_parallelism

  def _set_mutable(self, mutable):
    # [FUNCTION DOC]
    # HELPER: used to freeze or unfreeze options. Input: boolean. Sets the internal mutability
    # flag directly, bypassing the custom attribute setter. This class has no child option
    # objects.
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)


@tf_export("data.experimental.DistributeOptions")
class DistributeOptions(options_lib.OptionsBase):
  # [CLASS DOC]
  # Options for distributed data processing: the auto-shard policy (default automatic) and the
  # number of devices attached to the input pipeline (normally set by the multi-device
  # iterator). Converts to and from the distribute options protocol buffer message.
  """Represents options for distributed data processing.

  You can set the distribution options of a dataset through the
  `experimental_distribute` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.DistributeOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  dataset = dataset.with_options(options)
  ```
  """

  auto_shard_policy = options_lib.create_option(
      name="auto_shard_policy",
      ty=AutoShardPolicy,
      docstring="The type of sharding to use. See "
      "`tf.data.experimental.AutoShardPolicy` for additional information.",
      default_factory=lambda: AutoShardPolicy.AUTO)

  num_devices = options_lib.create_option(
      name="num_devices",
      ty=int,
      docstring=
      "The number of devices attached to this input pipeline. This will be "
      "automatically set by `MultiDeviceIterator`.")

  def _to_proto(self):
    # [FUNCTION DOC]
    # HELPER: serialization support. Returns a distribute options message with the auto-shard
    # policy always set (converted to its enum) and the device count only if set.
    pb = dataset_options_pb2.DistributeOptions()
    pb.auto_shard_policy = AutoShardPolicy._to_proto(self.auto_shard_policy)  # pylint: disable=protected-access
    if self.num_devices is not None:
      pb.num_devices = self.num_devices
    return pb

  def _from_proto(self, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support. Input: a distribute options message. Always sets the
    # auto-shard policy from it; sets the device count only if present. Returns nothing.
    self.auto_shard_policy = AutoShardPolicy._from_proto(pb.auto_shard_policy)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_num_devices") is not None:
      self.num_devices = pb.num_devices


@tf_export("data.experimental.OptimizationOptions")
class OptimizationOptions(options_lib.OptionsBase):
  # [CLASS DOC]
  # Options toggling static graph optimizations of the input pipeline: default optimizations,
  # filter fusion and parallelization, prefetch injection, sequential interleave prefetch, map
  # and batch fusion, map and filter fusion, map fusion, map parallelization, no-op elimination,
  # parallel batch, shuffle and repeat fusion. All default to unset. Converts to and from the
  # optimization options protocol buffer message.
  """Represents options for dataset optimizations.

  You can set the optimization options of a dataset through the
  `experimental_optimization` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.OptimizationOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_optimization.noop_elimination = True
  options.experimental_optimization.apply_default_optimizations = False
  dataset = dataset.with_options(options)
  ```
  """
  apply_default_optimizations = options_lib.create_option(
      name="apply_default_optimizations",
      ty=bool,
      docstring=
      "Whether to apply default graph optimizations. If False, only graph "
      "optimizations that have been explicitly enabled will be applied.")

  filter_fusion = options_lib.create_option(
      name="filter_fusion",
      ty=bool,
      docstring=
      "Whether to fuse filter transformations. If None, defaults to False.")

  filter_parallelization = options_lib.create_option(
      name="filter_parallelization",
      ty=bool,
      docstring=
      "Whether to parallelize stateless filter transformations. If None, "
      "defaults to False.")

  inject_prefetch = options_lib.create_option(
      name="inject_prefetch",
      ty=bool,
      docstring=
      "Whether to inject prefetch transformation as the last transformation "
      "when the last transformation is a synchronous transformation. If None, "
      "defaults to True.")

  seq_interleave_prefetch = options_lib.create_option(
      name="seq_interleave_prefetch",
      ty=bool,
      docstring=(
          "Whether to replace parallel interleave using a sequential interleave"
          " that prefetches elements from its input iterators. If None,"
          " defaults to False."
      ),
  )

  map_and_batch_fusion = options_lib.create_option(
      name="map_and_batch_fusion",
      ty=bool,
      docstring=
      "Whether to fuse map and batch transformations. If None, defaults to "
      "True.")

  map_and_filter_fusion = options_lib.create_option(
      name="map_and_filter_fusion",
      ty=bool,
      docstring=
      "Whether to fuse map and filter transformations. If None, defaults to "
      "False.")

  map_fusion = options_lib.create_option(
      name="map_fusion",
      ty=bool,
      docstring=(
          "Whether to fuse map transformations with `num_parallel_calls` set to"
          " `tf.data.AUTOTUNE`, no captured inputs and same `deterministic`"
          " value. If None, defaults to False."
      ),
  )

  map_parallelization = options_lib.create_option(
      name="map_parallelization",
      ty=bool,
      docstring=
      "Whether to parallelize stateless map transformations. If None, defaults "
      "to True.")

  noop_elimination = options_lib.create_option(
      name="noop_elimination",
      ty=bool,
      docstring=
      "Whether to eliminate no-op transformations. If None, defaults to True.")

  parallel_batch = options_lib.create_option(
      name="parallel_batch",
      ty=bool,
      docstring="Whether to parallelize copying of batch elements. If None, "
      "defaults to True.")

  shuffle_and_repeat_fusion = options_lib.create_option(
      name="shuffle_and_repeat_fusion",
      ty=bool,
      docstring="Whether to fuse shuffle and repeat transformations. If None, "
      "defaults to True.")

  def _to_proto(self):
    # [FUNCTION DOC]
    # HELPER: serialization support. Returns an optimization options message containing every
    # option that is set; unset options are left out.
    pb = dataset_options_pb2.OptimizationOptions()
    if self.apply_default_optimizations is not None:
      pb.apply_default_optimizations = self.apply_default_optimizations
    if self.filter_fusion is not None:
      pb.filter_fusion = self.filter_fusion
    if self.filter_parallelization is not None:
      pb.filter_parallelization = self.filter_parallelization
    if self.inject_prefetch is not None:
      pb.inject_prefetch = self.inject_prefetch
    if self.seq_interleave_prefetch is not None:
      pb.seq_interleave_prefetch = self.seq_interleave_prefetch
    if self.map_and_batch_fusion is not None:
      pb.map_and_batch_fusion = self.map_and_batch_fusion
    if self.map_and_filter_fusion is not None:
      pb.map_and_filter_fusion = self.map_and_filter_fusion
    if self.map_fusion is not None:
      pb.map_fusion = self.map_fusion
    if self.map_parallelization is not None:
      pb.map_parallelization = self.map_parallelization
    if self.noop_elimination is not None:
      pb.noop_elimination = self.noop_elimination
    if self.parallel_batch is not None:
      pb.parallel_batch = self.parallel_batch
    if self.shuffle_and_repeat_fusion is not None:
      pb.shuffle_and_repeat_fusion = self.shuffle_and_repeat_fusion
    return pb

  def _from_proto(self, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support. Input: an optimization options message. Sets on this
    # object each option whose optional field is present. Returns nothing.
    if pb.WhichOneof("optional_apply_default_optimizations") is not None:
      self.apply_default_optimizations = pb.apply_default_optimizations
    if pb.WhichOneof("optional_filter_fusion") is not None:
      self.filter_fusion = pb.filter_fusion
    if pb.WhichOneof("optional_filter_parallelization") is not None:
      self.filter_parallelization = pb.filter_parallelization
    if pb.WhichOneof("optional_inject_prefetch") is not None:
      self.inject_prefetch = pb.inject_prefetch
    if pb.WhichOneof("optional_seq_interleave_prefetch") is not None:
      self.seq_interleave_prefetch = pb.seq_interleave_prefetch
    if pb.WhichOneof("optional_map_and_batch_fusion") is not None:
      self.map_and_batch_fusion = pb.map_and_batch_fusion
    if pb.WhichOneof("optional_map_and_filter_fusion") is not None:
      self.map_and_filter_fusion = pb.map_and_filter_fusion
    if pb.WhichOneof("optional_map_fusion") is not None:
      self.map_fusion = pb.map_fusion
    if pb.WhichOneof("optional_map_parallelization") is not None:
      self.map_parallelization = pb.map_parallelization
    if pb.WhichOneof("optional_noop_elimination") is not None:
      self.noop_elimination = pb.noop_elimination
    if pb.WhichOneof("optional_parallel_batch") is not None:
      self.parallel_batch = pb.parallel_batch
    if pb.WhichOneof("optional_shuffle_and_repeat_fusion") is not None:
      self.shuffle_and_repeat_fusion = pb.shuffle_and_repeat_fusion

  def _set_mutable(self, mutable):
    # [FUNCTION DOC]
    # HELPER: used to freeze or unfreeze options. Input: boolean. Sets the internal mutability
    # flag directly, bypassing the custom attribute setter. No child option objects exist.
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)


@tf_export("data.experimental.ServiceOptions")
class ServiceOptions(options_lib.OptionsBase):
  # [CLASS DOC]
  # Options for the tf.data service client: a pinned-memory flag (default unset). Converts to
  # and from the service options protocol buffer message.
  """Represents options for tf.data service.

  You can set the service options of a dataset through the
  `experimental_service` property of `tf.data.Options`; the property is an
  instance of `tf.data.experimental.ServiceOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_service.pinned = True
  dataset = dataset.with_options(options)
  ```
  """

  pinned = options_lib.create_option(
      name="pinned",
      ty=bool,
      docstring=(
          "If true, the tf.data service client allocates data to pinned memory,"
          " which facilitates more efficient copying from host memory to GPU"
          " memory downstream. For gRPC, compression must be disabled for this"
          " to take effect. For alternative data transfer protocols, this may"
          " or may not take effect, depending on the implementation."
      ),
  )

  def _to_proto(self):
    # [FUNCTION DOC]
    # HELPER: serialization support. Returns a service options message with the pinned flag set
    # if it is set here.
    pb = dataset_options_pb2.ServiceOptions()
    if self.pinned is not None:
      pb.pinned = self.pinned
    return pb

  def _from_proto(self, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support. Input: a service options message. Sets the pinned flag if
    # present in it. Returns nothing.
    if pb.WhichOneof("optional_pinned") is not None:
      self.pinned = pb.pinned


@deprecation.deprecated_endpoints("data.experimental.ThreadingOptions")
@tf_export("data.experimental.ThreadingOptions", "data.ThreadingOptions")
class ThreadingOptions(options_lib.OptionsBase):
  # [CLASS DOC]
  # Options for dataset threading: maximum intra-op parallelism and private threadpool size
  # (zero means decide at run time from CPU count). Default unset. Converts to and from the
  # threading options protocol buffer message.
  """Represents options for dataset threading.

  You can set the threading options of a dataset through the
  `threading` property of `tf.data.Options`; the property is
  an instance of `tf.data.ThreadingOptions`.

  ```python
  options = tf.data.Options()
  options.threading.private_threadpool_size = 10
  dataset = dataset.with_options(options)
  ```
  """

  max_intra_op_parallelism = options_lib.create_option(
      name="max_intra_op_parallelism",
      ty=int,
      docstring=
      "If set, it overrides the maximum degree of intra-op parallelism.")

  private_threadpool_size = options_lib.create_option(
      name="private_threadpool_size",
      ty=int,
      docstring=
      "If set, the dataset will use a private threadpool of the given size. "
      "The value 0 can be used to indicate that the threadpool size should be "
      "determined at runtime based on the number of available CPU cores.")

  def _to_proto(self):
    # [FUNCTION DOC]
    # HELPER: serialization support. Returns a threading options message with each option that
    # is set copied in.
    pb = dataset_options_pb2.ThreadingOptions()
    if self.max_intra_op_parallelism is not None:
      pb.max_intra_op_parallelism = self.max_intra_op_parallelism
    if self.private_threadpool_size is not None:
      pb.private_threadpool_size = self.private_threadpool_size
    return pb

  def _from_proto(self, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support. Input: a threading options message. Sets each option whose
    # optional field is present. Returns nothing.
    if pb.WhichOneof("optional_max_intra_op_parallelism") is not None:
      self.max_intra_op_parallelism = pb.max_intra_op_parallelism
    if pb.WhichOneof("optional_private_threadpool_size") is not None:
      self.private_threadpool_size = pb.private_threadpool_size


@tf_export("data.Options")
class Options(options_lib.OptionsBase):
  # [CLASS DOC]
  # Top-level options object for a dataset. Holds deterministic ordering, external state policy,
  # slack, symbolic checkpointing, warm start, dataset name and framework type list, plus nested
  # autotune, distribute, optimization, service and threading option groups (created by default
  # factories). Deprecated aliases exist for the experimental deterministic and threading
  # options. Converts to and from the dataset options protocol buffer message, can propagate the
  # mutability flag to nested groups, and can be merged with another options object.
  """Represents options for `tf.data.Dataset`.

  A `tf.data.Options` object can be, for instance, used to control which static
  optimizations to apply to the input pipeline graph or whether to use
  performance modeling to dynamically tune the parallelism of operations such as
  `tf.data.Dataset.map` or `tf.data.Dataset.interleave`.

  The options are set for the entire dataset and are carried over to datasets
  created through tf.data transformations.

  The options can be set by constructing an `Options` object and using the
  `tf.data.Dataset.with_options(options)` transformation, which returns a
  dataset with the options set.

  >>> dataset = tf.data.Dataset.range(42)
  >>> options = tf.data.Options()
  >>> options.deterministic = False
  >>> dataset = dataset.with_options(options)
  >>> print(dataset.options().deterministic)
  False

  Note: A known limitation of the `tf.data.Options` implementation is that the
  options are not preserved across tf.function boundaries. In particular, to
  set options for a dataset that is iterated within a tf.function, the options
  need to be set within the same tf.function.
  """

  autotune = options_lib.create_option(
      name="autotune",
      ty=AutotuneOptions,
      docstring="The autotuning options associated with the dataset. See "
      "`tf.data.experimental.AutotuneOptions` for more details.",
      default_factory=AutotuneOptions)

  deterministic = options_lib.create_option(
      name="deterministic",
      ty=bool,
      docstring=
      "Whether the outputs need to be produced in deterministic order. If None,"
      " defaults to True.")

  experimental_deterministic = options_lib.create_option(
      name="experimental_deterministic",
      ty=bool,
      docstring="DEPRECATED. Use `deterministic` instead.")

  experimental_distribute = options_lib.create_option(
      name="experimental_distribute",
      ty=DistributeOptions,
      docstring=
      "The distribution strategy options associated with the dataset. See "
      "`tf.data.experimental.DistributeOptions` for more details.",
      default_factory=DistributeOptions)

  experimental_external_state_policy = options_lib.create_option(
      name="experimental_external_state_policy",
      ty=ExternalStatePolicy,
      docstring="This option can be used to override the default policy for "
      "how to handle external state when serializing a dataset or "
      "checkpointing its iterator. There are three settings available - "
      "IGNORE: External state is ignored without a warning; WARN: External "
      "state is ignored and a warning is logged; FAIL: External state results "
      "in an error.")

  experimental_optimization = options_lib.create_option(
      name="experimental_optimization",
      ty=OptimizationOptions,
      docstring=
      "The optimization options associated with the dataset. See "
      "`tf.data.experimental.OptimizationOptions` for more details.",
      default_factory=OptimizationOptions)

  experimental_slack = options_lib.create_option(
      name="experimental_slack",
      ty=bool,
      docstring="Whether to introduce 'slack' in the last `prefetch` of the "
      "input pipeline, if it exists. This may reduce CPU contention with "
      "accelerator host-side activity at the start of a step. The slack "
      "frequency is determined by the number of devices attached to this "
      "input pipeline. If None, defaults to False.")

  experimental_symbolic_checkpoint = options_lib.create_option(
      name="experimental_symbolic_checkpoint",
      ty=bool,
      docstring="Whether to checkpoint internal input pipeline state "
      "maintaining cursors into data sources that identify last "
      "element(s) produced as output to the tf.data consumer. This "
      "is alternative to the default 'explicit' checkpointing which "
      "stores the internal input pipeline state in the checkpoint. "
      "Note that symbolic checkpointing is not supported for "
      "transformations that can reorder elements.")

  experimental_service = options_lib.create_option(
      name="experimental_service",
      ty=ServiceOptions,
      docstring=(
          "The tf.data service options associated with the dataset. See "
          "`tf.data.experimental.ServiceOptions` for more details."
      ),
      default_factory=ServiceOptions,
  )

  experimental_threading = options_lib.create_option(
      name="experimental_threading",
      ty=ThreadingOptions,
      docstring="DEPRECATED. Use `threading` instead.")

  experimental_warm_start = options_lib.create_option(
      name="experimental_warm_start",
      ty=bool,
      docstring=(
          "Whether to start background threads of asynchronous transformations "
          "upon iterator creation, as opposed to during the first call to "
          "`next()`. Defaults to `False`. "
          "This improves the latency of the initial 'next()' calls at "
          "the expense of requiring more memory to hold prefetched elements "
          "between the time of iterator construction and usage."
      ),
      default_factory=lambda: True if test_mode.TEST_MODE else None,
  )

  dataset_name = options_lib.create_option(
      name="dataset_name",
      ty=str,
      docstring="A name for the dataset, to help in debugging.")

  framework_type = options_lib.create_option(
      name="framework_type",
      ty=list,
      docstring="The list of frameworks that are used to generate this "
      "pipeline, used for telemetry.")

  threading = options_lib.create_option(
      name="threading",
      ty=ThreadingOptions,
      docstring="The threading options associated with the dataset. See "
      "`tf.data.ThreadingOptions` for more details.",
      default_factory=ThreadingOptions)

  def __getattribute__(self, name):
    # [FUNCTION DOC]
    # Attribute read hook that redirects the two deprecated names. Reading the deprecated
    # threading option logs a deprecation warning and returns the threading option group.
    # Reading the deprecated deterministic option returns the deterministic option without
    # warning (the warning is commented out). Any other name uses the normal lookup.
    if name == "experimental_threading":
      logging.warning("options.experimental_threading is deprecated. "
                      "Use options.threading instead.")
      return getattr(self, "threading")
    if name == "experimental_deterministic":
      # TODO(aaudibert): Uncomment after internal uses have been updated.
      # logging.warning("options.experimental_deterministic is deprecated. "
      #                 "Use options.deterministic instead.")
      return getattr(self, "deterministic")
    return super(Options, self).__getattribute__(name)

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Attribute write hook. Setting the deprecated threading name logs a warning and sets the
    # threading option group instead. Setting the deprecated deterministic name sets the
    # deterministic option (no warning). Setting symbolic checkpointing on macOS only logs a
    # warning that it is unsupported and silently does not store the value. Any other name is
    # stored through the base class setter (which enforces mutability).
    if name == "experimental_threading":
      logging.warning("options.experimental_threading is deprecated. "
                      "Use options.threading instead.")
      super(Options, self).__setattr__("threading", value)
      return
    if name == "experimental_deterministic":
      # TODO(aaudibert): Uncomment after internal uses have been updated.
      # logging.warning("options.experimental_deterministic is deprecated. "
      #                 "Use options.deterministic instead.")
      super(Options, self).__setattr__("deterministic", value)
      return
    if name == "experimental_symbolic_checkpoint":
      # TODO(b/276269493): Add support for MacOS.
      if platform.system() == "Darwin":
        logging.warning("Symbolic checkpointing is not supported on MacOS.")
        return
    super(Options, self).__setattr__(name, value)

  def _to_proto(self):
    # [FUNCTION DOC]
    # HELPER: serialization support. Returns an options message with each set scalar option
    # copied in (deterministic, external state policy converted to its enum, slack, symbolic
    # checkpoint, warm start, dataset name, framework types appended one by one) and with the
    # nested autotune, distribute, optimization, service and threading messages always copied in.
    pb = dataset_options_pb2.Options()
    if self.deterministic is not None:
      pb.deterministic = self.deterministic
    pb.autotune_options.CopyFrom(self.autotune._to_proto())  # pylint: disable=protected-access
    pb.distribute_options.CopyFrom(self.experimental_distribute._to_proto())  # pylint: disable=protected-access
    if self.experimental_external_state_policy is not None:
      pb.external_state_policy = (
          ExternalStatePolicy._to_proto(  # pylint: disable=protected-access
              self.experimental_external_state_policy))
    pb.optimization_options.CopyFrom(self.experimental_optimization._to_proto())  # pylint: disable=protected-access
    if self.experimental_slack is not None:
      pb.slack = self.experimental_slack
    if self.experimental_symbolic_checkpoint is not None:
      pb.symbolic_checkpoint = self.experimental_symbolic_checkpoint
    if self.experimental_warm_start is not None:
      pb.warm_start = self.experimental_warm_start
    if self.dataset_name is not None:
      pb.dataset_name = self.dataset_name
    if self.framework_type:
      for framework_type in self.framework_type:
        pb.framework_type.append(framework_type)
    pb.service_options.CopyFrom(self.experimental_service._to_proto())  # pylint: disable=protected-access
    pb.threading_options.CopyFrom(self.threading._to_proto())  # pylint: disable=protected-access
    return pb

  def _from_proto(self, pb):
    # [FUNCTION DOC]
    # HELPER: deserialization support. Input: an options message. Sets each scalar option whose
    # optional field is present, replaces the framework type list when the message has entries,
    # and loads every nested option group from its sub-message. Returns nothing.
    if pb.WhichOneof("optional_deterministic") is not None:
      self.deterministic = pb.deterministic
    self.autotune._from_proto(pb.autotune_options)  # pylint: disable=protected-access
    self.experimental_distribute._from_proto(pb.distribute_options)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_external_state_policy") is not None:
      self.experimental_external_state_policy = (
          ExternalStatePolicy._from_proto(  # pylint: disable=protected-access
              pb.external_state_policy))
    self.experimental_optimization._from_proto(pb.optimization_options)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_slack") is not None:
      self.experimental_slack = pb.slack
    if pb.WhichOneof("optional_symbolic_checkpoint") is not None:
      self.experimental_symbolic_checkpoint = pb.symbolic_checkpoint
    if pb.WhichOneof("optional_warm_start") is not None:
      self.experimental_warm_start = pb.warm_start
    if pb.WhichOneof("optional_dataset_name") is not None:
      self.dataset_name = pb.dataset_name
    if pb.framework_type:
      self.framework_type = []
      for framework_type in pb.framework_type:
        self.framework_type.append(framework_type)
    self.experimental_service._from_proto(pb.service_options)  # pylint: disable=protected-access
    self.threading._from_proto(pb.threading_options)  # pylint: disable=protected-access

  def _set_mutable(self, mutable):
    # [FUNCTION DOC]
    # HELPER: used to freeze or unfreeze options. Input: boolean. Sets the internal mutability
    # flag on this object directly and propagates it to the autotune, distribute, optimization
    # and threading groups; the service group is not updated.
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
    self.autotune._set_mutable(mutable)
    self.experimental_distribute._set_mutable(mutable)
    self.experimental_optimization._set_mutable(mutable)
    self.threading._set_mutable(mutable)

  def merge(self, options):
    # [FUNCTION DOC]
    # Input: another options object. Delegates to the shared merge utility, which combines both
    # objects option by option, warns on conflicting values and returns a new options object.
    """Merges itself with the given `tf.data.Options`.

    If this object and the `options` to merge set an option differently, a
    warning is generated and this object's value is updated with the `options`
    object's value.

    Args:
      options: The `tf.data.Options` to merge with.

    Returns:
      New `tf.data.Options` object which is the result of merging self with
      the input `tf.data.Options`.
    """
    return options_lib.merge_options(self, options)
