# [MODULE DOC]
# Library of TPU helper functions (the core of the v1 TPU API). Provides system initialization and
# shutdown ops, the replicate/shard/batch_parallel/rewrite entry points that wrap a computation in
# a TPU cluster context, the lower level split compile-and-replicate and compile-and-shard
# builders, dynamic-shape padding of replica inputs, output post-processing, and graph rewriting
# for TPU inference. Also defines the padding policy enumeration and the XLA options tuple.
# Import-time side effects: registers a resource resolver with the automatic-control-dependency
# machinery, and registers public symbols through the export decorators. Defines module-level
# attribute names used to mark ops for post-device rewrite and compilation status.
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
# ======================================

"""Library of TPU helper functions."""

import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union

from absl import logging
import numpy as np

from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export


# Ops which can be safely pruned from XLA compile if they have no consumers.
#  These ops should also have no inputs.
_UNCONNECTED_OPS_TO_PRUNE = set(["Placeholder", "VarHandleOp"])

_POST_DEVICE_REWRITE_ATTR = "_post_device_rewrite"
_TPU_COMPILATION_STATUS_ATTR = "_tpu_compilation_status"
_PIVOT_FOR_CLUSTER = "_pivot_for_cluster"


core = tpu_name_util.core


def _tpu_system_device_name(job: Optional[Text]) -> Text:
  # [FUNCTION DOC]
  # HELPER: builds device strings for the TPU system ops in this module.
  # Input: job, an optional job name. Returns the device name of the TPU system device on task
  # zero: job-qualified when a job is given, otherwise the unqualified default form. No side
  # effects, no exceptions.
  """Returns the device name for the TPU_SYSTEM device of `job`."""
  if job is None:
    return "/device:TPU_SYSTEM:0"
  else:
    return "/job:%s/device:TPU_SYSTEM:0" % job


@tf_export(v1=["tpu.initialize_system"])
def initialize_system(
    embedding_config: Optional[embedding_pb2.TPUEmbeddingConfiguration] = None,
    job: Optional[Text] = None,
    compilation_failure_closes_chips: bool = True,
    tpu_cancellation_closes_chips: Optional[bool] = None,
) -> core_types.Tensor:
  # [FUNCTION DOC]
  # Builds the graph op that initializes the distributed TPU system and yields its topology.
  # Inputs: embedding_config (optional embedding configuration proto), job (optional job name
  # holding the TPU devices), compilation_failure_closes_chips (default true) and
  # tpu_cancellation_closes_chips (optional; nothing means the runtime flag decides, true or false
  # maps to an explicit setting). Placed on the TPU system device of the job. Without an embedding
  # config returns the configure op output directly; with one, the embedding configuration op runs
  # after the topology op and the returned tensor is an identity of the topology that depends on
  # it. Returns a tensor holding the serialized topology, which must be evaluated in a session.
  # This only builds graph ops; the system is initialized when they are run.
  """Initializes a distributed TPU system for use with TensorFlow.

  Args:
    embedding_config: If not None, a `TPUEmbeddingConfiguration` proto
      describing the desired configuration of the hardware embedding lookup
      tables. If embedding_config is None, no hardware embeddings can be used.
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be initialized. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
    compilation_failure_closes_chips: Set the configuration whether
      we want to close TPU chips when there is a compilation failure.
    tpu_cancellation_closes_chips: Set the configuration whether
      we want to close TPU chips when a TPU execution is cancelled. If the value
      is None, the behavior will be determined by the command line flag
      `tpu_cancellation_closes_chips` for the TPU worker. WARNING: this argument
      only applies to TFRT TPU runtime.
  Returns:
    A serialized `TopologyProto` that describes the TPU system. Note:
      the topology must be evaluated using `Session.run` before it can be used.
  """
  config_string = ("" if embedding_config is None else
                   embedding_config.SerializeToString())

  # The enum is defined in core/tpu/kernels/tpu_execute_op_options.h.
  tpu_cancellation_closes_chips_enum = 0
  if tpu_cancellation_closes_chips is not None:
    if tpu_cancellation_closes_chips:
      tpu_cancellation_closes_chips_enum = 1
    else:
      tpu_cancellation_closes_chips_enum = 2

  with ops.device(_tpu_system_device_name(job)):
    topology = tpu_ops.configure_distributed_tpu(
        compilation_failure_closes_chips=compilation_failure_closes_chips,
        tpu_cancellation_closes_chips=tpu_cancellation_closes_chips_enum,
    )

    if embedding_config is None:
      return topology

    # This set of control dependencies is needed as this function is expected to
    # return an op which will return the topology when executed, but we need to
    # call the embedding initialization op between initializing the TPU and
    # returning the topology.
    with ops.control_dependencies([topology]):
      embedding_init = tpu_ops.configure_tpu_embedding(config=config_string)
    with ops.control_dependencies([embedding_init]):
      return array_ops.identity(topology, name="tpu_init_identity")


def initialize_system_for_tpu_embedding(
    embedding_config: embedding_pb2.TPUEmbeddingConfiguration,
    job: Optional[Text] = None,
) -> ops.Operation:
  # [FUNCTION DOC]
  # Builds the op that configures TPU embedding hardware on an already initialized TPU system.
  # Inputs: embedding_config (embedding configuration proto, serialized here) and job (optional
  # job name). The op is placed on the job's TPU system device. Returns that operation.
  # Do not combine with initialize_system called with an embedding config.
  """Initializes a distributed TPU Embedding system for use with TensorFlow.

  The following two are equivalent:
  1. initialize_system() with embedding_config.
  2. initialize_system() without embedding_config, then
     initialize_system_for_tpu_embedding().
  initialize_system() should not be called with embedding_config if
  initialize_system_for_tpu_embedding() is meant to be called later.

  Args:
    embedding_config: a `TPUEmbeddingConfiguration` proto describing the desired
      configuration of the hardware embedding lookup tables.
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be initialized. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.

  Returns:
    A no-op.
  """
  config_string = embedding_config.SerializeToString()
  with ops.device(_tpu_system_device_name(job)):
    return tpu_ops.configure_tpu_embedding(config=config_string)


@tf_export(v1=["tpu.shutdown_system"])
def shutdown_system(job: Optional[Text] = None) -> ops.Operation:
  # [FUNCTION DOC]
  # Builds the op that shuts down the distributed TPU system. Input: job, optional job name used to
  # pick the TPU system device. Returns the shutdown operation (the docstring omits this); the
  # system is only shut down when the op is run.
  """Shuts down a running a distributed TPU system.

  Args:
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be shutdown. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
  """
  with ops.device(_tpu_system_device_name(job)):
    shutdown_distributed_tpu = tpu_ops.shutdown_distributed_tpu()
  return shutdown_distributed_tpu


@auto_control_deps.register_acd_resource_resolver
def tpu_replicated_input_resolver(
    op: ops.Operation,
    resource_reads: object_identity.ObjectIdentitySet,
    resource_writes: object_identity.ObjectIdentitySet) -> bool:
  # [FUNCTION DOC]
  # HELPER: resolver registered with automatic control dependencies for TPU replicated inputs.
  # Inputs: op, plus the sets of resources it reads and writes (mutated in place). If the op is a
  # replicated-input op, both sets are cleared (control dependencies are added on its replicated
  # inputs instead) and it returns true only if either set was non-empty. For other ops, each
  # resource that is an output of a replicated-input op is replaced in the sets by the underlying
  # replicated inputs. Returns true if any set was changed, false otherwise.
  """Replaces TPUReplicatedInput outputs with its inputs in resource_inputs."""
  # Ignore TPUReplicatedInput for ACD purposes since we will be directly adding
  # control deps on the replicated inputs.
  if op.type == "TPUReplicatedInput":
    if resource_reads or resource_writes:
      resource_reads.clear()
      resource_writes.clear()
      return True
    else:
      return False
  # Replace tensors in `resource_inputs` which are outputs of TPUReplicatedInput
  # with the actual replicated inputs. This allows ACD to correct add control
  # deps when there are multiple calls to `run` in a
  # `tf.function`.
  def replace_with_unreplicated_resources(resource_inputs):
    # [FUNCTION DOC]
    # HELPER: rewrites one resource set in place. Removes every resource produced by a
    # replicated-input op and adds that op's inputs instead. Input: a mutable set of resources.
    # Returns the (truthy) list of added resources, or else the list of removed ones; both are
    # empty if nothing changed.
    """Replaces handles in `resource_inputs` with their unreplicated inputs."""
    to_remove = []
    to_add = []
    for resource in resource_inputs:
      if resource.op.type == "TPUReplicatedInput":
        to_remove.append(resource)
        to_add.extend(resource.op.inputs)
    for t in to_remove:
      resource_inputs.discard(t)
    resource_inputs.update(to_add)
    return to_add or to_remove

  return bool(replace_with_unreplicated_resources(resource_reads) or
              replace_with_unreplicated_resources(resource_writes))


@tf_export(v1=["tpu.PaddingSpec"])
class PaddingSpec(enum.IntEnum):
  # [CLASS DOC]
  # Integer enumeration of padding policies for dynamic input shapes in replicate. AUTO (0) pads a
  # dynamic dimension to the maximum over all replicas; POWER_OF_TWO (1) additionally rounds that
  # size up to a power of two to limit recompilation. Exported under the v1 TPU namespace.
  """Represents the type of padding policies for tpu.replicate."""
  # By default the policy is set to AUTO, the dynamic input shape dimension will
  # be pad to maximum of all the replicas.
  AUTO = 0
  # Bucketize the dynamic input shape dimension into a power of 2.
  POWER_OF_TWO = 1


@tf_export("tpu.XLAOptions")
class XLAOptions(
    collections.namedtuple(
        "XLAOptions",
        [
            "use_spmd_for_xla_partitioning",
            "enable_xla_dynamic_padder",
            "use_shardy_partitioner",
        ],
    )
):
  # [CLASS DOC]
  # Immutable named tuple of options passed to the XLA compiler for TPU computations, with fields
  # for using the SPMD partitioner, enabling the XLA dynamic padder and using the Shardy
  # partitioner. Created by users and passed to replicate and related functions. Exported publicly.
  """XLA compilation options.

  Attributes:
    use_spmd_for_xla_partitioning: Boolean. Whether to use XLA's SPMD
      partitioner instead of MPMD partitioner when compiler partitioning is
      requested.
    enable_xla_dynamic_padder: Boolean. Whether to enable XLA dynamic padder
      infrastructure to handle dynamic shapes inputs inside XLA. True by
      default. Disabling this may cause correctness issues with dynamic shapes
      inputs, as XLA will just assume the inputs are with padded shapes. However
      users can optionally set it to False to improve device time if masking is
      already handled in the user side.
    use_shardy_partitioner: Boolean. Whether to use Shardy partitioner.
  """

  def __new__(
      cls,
      use_spmd_for_xla_partitioning=True,
      enable_xla_dynamic_padder=True,
      use_shardy_partitioner=False,
  ):
    # [FUNCTION DOC]
    # Tuple constructor supplying defaults. Inputs: use_spmd_for_xla_partitioning (default true),
    # enable_xla_dynamic_padder (default true) and use_shardy_partitioner (default false), all
    # booleans. Returns the new options tuple; no side effects.
    return super(XLAOptions, cls).__new__(
        cls,
        use_spmd_for_xla_partitioning,
        enable_xla_dynamic_padder,
        use_shardy_partitioner,
    )


@tf_export(v1=["tpu.replicate"])
@traceback_utils.filter_traceback
def replicate(
    computation: Callable[..., Any],
    inputs: Optional[List[List[core_types.Tensor]]] = None,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[device_assignment_lib.DeviceAssignment] = None,
    name: Optional[Text] = None,
    maximum_shapes: Optional[Any] = None,
    padding_spec: Optional[PaddingSpec] = None,
    xla_options: Optional[XLAOptions] = None) -> List[Any]:
  # [FUNCTION DOC]
  # Public wrapper that builds the replicated TPU computation and returns only the outputs. It
  # forwards all arguments (computation, per-replica inputs, infeed queue, device assignment,
  # deprecated name, maximum shapes, padding spec, XLA options) to the split compile-and-replicate
  # builder and returns the second element of its result: the per-replica outputs. Exceptions
  # (value errors for mismatched input counts or shapes) come from the builder. Traceback
  # filtering is applied by decorator.
  """Builds a graph operator that runs a replicated TPU computation.

  Example for the basic usage that `inputs` has static shape:

  ```python

  def computation(x):
    x = x + 1
    return tf.math.reduce_mean(x)

  x = tf.convert_to_tensor([1., 2., 3.])
  y = tf.convert_to_tensor([4., 5., 6.])
  tf.compat.v1.tpu.replicate(computation, inputs=[[x], [y]])
  ```

  If the `inputs` has dynamic shapes and you would like to automatically
  bucketize the inputs to avoid XLA recompilation. See the advanced example
  below:

  ```python

  def computation(x):
    x = x + 1
    return tf.math.reduce_mean(x)

  # Assume input tensors in two replicas `x` and `y` both have dynamic shape
  # ([None, 2]).
  tf.compat.v1.tpu.replicate(
    computation,
    inputs=[x, y],
    maximum_shapes=[tf.TensorShape([None, None])],
    padding_spec=tf.compat.v1.tpu.PaddingSpec.POWER_OF_TWO)
  ```

  Args:
    computation: A Python function that builds the computation to replicate.
    inputs: A list of lists of input tensors or `None` (equivalent to
      `[[]]`), indexed by `[replica_num][input_num]`. All replicas must
      have the same number of inputs. Each input can be a nested structure
      containing values that are convertible to tensors. Note that passing an
      N-dimension list of compatible values will result in a N-dimension list of
      scalar tensors rather than a single Rank-N tensors. If you need different
      behavior, convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to computation.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each replica of the computation uses
      only one core, and there is either only one replica, or the number of
      replicas is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    maximum_shapes: A nested structure of tf.TensorShape representing the shape
      to which the respective component of each input element in each replica
      should be padded. Any unknown dimensions (e.g.
      tf.compat.v1.Dimension(None) in a tf.TensorShape or -1 in a tensor-like
      object) will be padded to the maximum size of that dimension over all
      replicas. The structure of `maximum_shapes` needs to be the same as
      `inputs[0]`.
    padding_spec: An enum specified by `tpu.PaddingSpec`. This describes the
      padding policy when the `inputs` to `tpu.replicate` is dynamic.
      One usage is to enable automatic bucketizing on the inputs by setting the
      value to `tpu.PaddingSpec.POWER_OF_TWO`, which can help to reduce the
      recompilation in the XLA side.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.
  Returns:
    A list of outputs, indexed by `[replica_num]` each output can be a nested
    structure same as what computation() returns with a few exceptions.

    Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
      TODO(b/121383831): Investigate into removing these special cases.

  Raises:
    ValueError: If all replicas do not have equal numbers of input tensors.
    ValueError: If the number of inputs per replica does not match
      the number of formal parameters to `computation`.
    ValueError: If the static `inputs` dimensions don't match with the values
      given in `maximum_shapes`.
    ValueError: If the structure of inputs per replica does not match
      the structure of `maximum_shapes`.
  """
  return split_compile_and_replicate(
      computation,
      inputs,
      infeed_queue,
      device_assignment,
      name,
      maximum_shapes=maximum_shapes,
      padding_spec=padding_spec,
      xla_options=xla_options)[1]


def _ceil_to_pow_of_n(x, n):
  # [FUNCTION DOC]
  # HELPER: rounding for power-of-two bucketing of dynamic dimensions during padding.
  # Inputs: x (integer tensor or value) and n (base). Computes, via floating point logarithms, the
  # smallest power of n that is at least x. Returns a 32-bit integer tensor. Because it uses
  # float arithmetic it may be imprecise for very large values.
  """Ceil input `x` to power of `n`."""
  x = math_ops.cast(x, dtypes.float32)
  lognx = math_ops.log(x) / math_ops.log(n * 1.0)
  lognx = math_ops.ceil(lognx)
  result = math_ops.pow(n * 1.0, lognx)
  result = math_ops.cast(result, dtypes.int32)
  return result


def _pad_all_input(
    inputs: Iterable[core_types.Tensor],
    padded_shapes: List[Optional[tensor_shape.TensorShape]],
    padding_spec: PaddingSpec
) -> Tuple[List[List[Any]], List[dynamic_padding.PaddingMap]]:
  # [FUNCTION DOC]
  # HELPER: dynamic-shape preprocessing step for split_compile_and_replicate.
  # Inputs: inputs (per replica lists of tensors), padded_shapes (per input position, a target
  # shape or nothing to skip padding) and padding_spec (automatic or power-of-two policy).
  # Finds, per input position, which dimensions differ or are unknown across replicas; for those
  # it pads each replica's tensor up to the given maximum, or the cross-replica maximum of the
  # runtime shape (at least 2 because XLA does not support dynamic size 1), rounded to a power of
  # two if requested. Real shape values are appended to each replica's inputs, and shape and
  # padded ops are marked with the post-device-rewrite attribute (mutating the graph ops). Fully
  # static inputs are padded under a trivially true conditional so shapes stay dynamic.
  # Returns the padded inputs per replica and the list of padding maps linking padded dimensions
  # to the arguments that carry their real size.
  """Pad all input tensors given padded_shapes.

  The real shape tensors will be concatenated with the padded original inputs.

  Args:
    inputs: The original inputs.
    padded_shapes: A list of padded shapes for each input. If an entry is None,
      no padding is performed.
    padding_spec: An enum specified by `tpu.PaddingSpec`. This describes the
      padding policy when the `inputs` to `tf.tpu.replicate` is dynamic.
      One usage is to enable automatic bucketizing on the inputs by setting the
      value to `tpu.PaddingSpec.POWER_OF_TWO`, which can help to reduce the
      recompilation in the XLA side.

  Returns:
    The padded inputs and a PaddingMap list which maps the padded input
    dimension to the real shape argument index.
  """
  # maximum_static_shapes[idx][i] indicates the maximum static size of ith
  # dimension of the idx input among all the replicas.
  maximum_static_shapes = []
  # need_padding[idx][i] indicates whether the ith dimension of the idx input
  # needs padding.
  need_padding = []
  input_shape_tensors = []
  for core_idx, inputs_per_core in enumerate(inputs):
    for idx, input_tensor in enumerate(inputs_per_core):
      input_shape = input_tensor.get_shape().as_list()
      if core_idx == 0:
        input_shape_tensors.append([])
        maximum_static_shapes.append(input_shape)
        need_padding.append(np.full_like(input_shape, False, dtype=bool))
      else:
        for i, s in enumerate(input_shape):
          if s is None or s != maximum_static_shapes[idx][i]:
            need_padding[idx][i] = True
        maximum_static_shapes[idx] = max(input_shape,
                                         maximum_static_shapes[idx])

      # Append _POST_DEVICE_REWRITE_ATTR attributes to the real shape ops.
      real_input_shape = array_ops.shape(input_tensor)
      real_input_shape.op._set_attr(  # pylint: disable=protected-access
          _POST_DEVICE_REWRITE_ATTR,
          attr_value_pb2.AttrValue(b=True))
      input_shape_tensors[idx].append(real_input_shape)

  maximum_shapes = []
  for shapes_per_input in input_shape_tensors:
    maximum_shapes.append(
        math_ops.reduce_max(array_ops_stack.stack(shapes_per_input), axis=0))

  padded_inputs = []
  real_shapes = []
  padding_maps = []
  for core_idx, inputs_per_core in enumerate(inputs):
    padded_inputs.append([])
    real_shapes.append([])
    real_shape_idx = len(inputs_per_core) - 1
    for idx, input_tensor in enumerate(inputs_per_core):
      input_shape_tensor = input_shape_tensors[idx][core_idx]
      input_shape = input_tensor.get_shape().as_list()
      padded_shape = padded_shapes[idx]

      # If we have no padded_shape, then skip padding.
      if any(need_padding[idx]) and padded_shape is not None:
        for i, s in enumerate(input_shape):
          if need_padding[idx][i]:
            if core_idx == 0:
              real_shape_idx += 1
              padding_map = dynamic_padding.PaddingMap()
              padding_map.arg_index = idx
              padding_map.shape_index = i
              padding_map.padding_arg_index = real_shape_idx
              padding_maps.append(padding_map)
            real_shapes[core_idx].append(
                math_ops.cast(input_shape_tensor[i], dtypes.int32))

        paddings = []
        for i, s in enumerate(padded_shape.dims):
          if need_padding[idx][i]:
            # The minimum padded dimension size is 2 as XLA doesn't support size
            # 1 dynamic size.
            minimum_dynamic_dim_size = 2
            if s.value is not None:
              # Pad to the given maximum value.
              max_dim_size = max(s.value, minimum_dynamic_dim_size)
            else:
              # If maximum value is not given, then pad to the maximum dimension
              # among all the cores.
              max_dim_size = math_ops.maximum(maximum_shapes[idx][i],
                                              minimum_dynamic_dim_size)
              if padding_spec == PaddingSpec.POWER_OF_TWO:
                max_dim_size = _ceil_to_pow_of_n(max_dim_size, 2)
            # Pad to the given maximum value.
            padding = [0, max_dim_size - input_shape_tensor[i]]
          else:
            padding = [0, 0]
          paddings.append(padding)

        if input_tensor.get_shape().is_fully_defined():
          # TODO(rxsang): This is a hack to make sure padded_input has dynamic
          # shapes, so any tf.size/tf.shape op performed on it won't be constant
          # folded. Do we have better ways to do it?
          padded_input = cond.cond(
              array_ops.constant(True),
              lambda: array_ops.pad(input_tensor, paddings),  # pylint: disable=cell-var-from-loop
              lambda: input_tensor)
        else:
          padded_input = array_ops.pad(input_tensor, paddings)

        # Append _POST_DEVICE_REWRITE_ATTR attributes to all padded inputs.
        padded_input.op._set_attr(  # pylint: disable=protected-access
            _POST_DEVICE_REWRITE_ATTR,
            attr_value_pb2.AttrValue(b=True))

        padded_inputs[core_idx].append(padded_input)
      else:
        padded_inputs[core_idx].append(input_tensor)

  num_replicas = len(padded_inputs)
  for i in range(num_replicas):
    padded_inputs[i].extend(real_shapes[i])

  return padded_inputs, padding_maps


def _flatten_and_filter_composite(maybe_composite, non_composite_output,
                                  composite_output=None):
  # [FUNCTION DOC]
  # HELPER: builds masks that mark which flattened components came from composite tensors.
  # Inputs: maybe_composite (any value), non_composite_output (returned for non-composites) and
  # composite_output (default nothing; repeated once per flattened component for a composite).
  # Returns a tuple of repeated composite_output values with one entry per component when the
  # value is a composite tensor, otherwise non_composite_output unchanged.
  """For an input, replaced the input by a tuple if the input is composite.

  If `maybe_composite` is not composite, return the parameter
  `non_composite_output` otherwise return a tuple which consists of the value of
  the parameter `composite_output` the same number of times as there are
  components of the composite tensor.

  This is useful for computing a mask when flattening nested data with
  `expand_composites=True`. For example

  ```python
  nest.flatten(data, expand_composites=True)
  ```

  and

  ```python
  nest.flatten(nest.map(
      data, lambda x: _flatten_and_filter_composite(x, False, True)))
  ```

  will have the same length and second will be True if the tensor in the first
  is derived from a expanding a composite tensor.

  Args:
    maybe_composite: A value to test for being a composite tensor.
    non_composite_output: The value to return when `maybe_composite` is not a
      composite.
    composite_output: the value to fill the output tuple with if
      `maybe_composite` is a composite.

  Returns:
    `non_composite_output` or a tuple with multiple copies of
    `composite_output`.
  """

  if isinstance(maybe_composite, composite_tensor.CompositeTensor):
    num_components = len(nest.flatten(maybe_composite, expand_composites=True))
    return (composite_output,) * num_components
  return non_composite_output


def split_compile_and_replicate(
    computation: Callable[..., Any],
    inputs: Optional[List[List[core_types.Tensor]]] = None,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[device_assignment_lib.DeviceAssignment] = None,
    name: Optional[Text] = None,
    use_tpu: bool = True,
    maximum_shapes: Optional[Any] = None,
    padding_spec: Optional[PaddingSpec] = None,
    xla_options: Optional[XLAOptions] = None,
) -> List[List[core_types.Tensor]]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the return annotation claims a list of lists of tensors, but the real result is a
  # two element list (compile status, then per-replica outputs). With zero replicas it returns an
  # empty list, so the public replicate wrapper, which takes the second element, would then fail
  # with an index error. The docstring lists only value errors, but type errors are also raised;
  # in the infeed-queue case the error is built with a stray comma so its message is a tuple.
  # Core builder behind replicate. Inputs: computation (callable building the per-replica work),
  # inputs (list of per-replica lists or tuples of values, nothing meaning one replica with no
  # inputs), infeed_queue, device_assignment, deprecated name (deleted), use_tpu (default true;
  # false runs on XLA CPU or GPU), maximum_shapes and padding_spec (dynamic shape padding) and
  # xla_options (default options when nothing).
  # Steps: builds replicate metadata (topology and device assignment, soft placement flag, step
  # marker location, partitioner settings); validates input structure, counts and types across
  # replicas; converts variables to tensors and substitutes a placeholder for nothing values;
  # checks the computation accepts the arguments; optionally pads dynamic shapes; fans inputs in
  # with replicated-input ops inside a new cluster context (pivot op, unique cluster name);
  # sets dynamic dimension sizes; wraps inputs in identity ops; appends infeed dequeue values;
  # runs the computation with resource variables forced on and partitioners disabled (temporarily
  # replacing the variable scope's settings, restored afterwards); post-processes flat or nested
  # outputs; optionally traces tensors with the tensor tracer; and fans outputs out per replica.
  # Always reports unsupported operations and exits the context. Returns a two element list: the
  # compilation status tensor or no-op, and per-replica outputs in the structure the computation
  # returned (or grouped no-ops when it produced no tensors).
  # Raises a type error if inputs is not a list of lists or tuples or the computation cannot take
  # the arguments; a value error for unequal input counts or types, or dynamic shapes with infeed.
  # Side effects: mutates the default graph, variable scope and ops attributes, and logs messages.
  """Builds graph operators that runs compilation and replicated computation.

  This is a lower level interface than replicate that returns a separate compile
  and execute output tensor. In the generated graph the compile op feeds into
  the execute op and no additional compilation is incurred when running the
  compile op before the execute op. The compile op returns additional
  information about the compilation but does not return the compiled program.

  Args:
    computation: A Python function that builds the computation to replicate.
    inputs: A list of lists of input tensors or `None` (equivalent to
      `[[]]`), indexed by `[replica_num][input_num]`. All replicas must
      have the same number of inputs. Each input can be a nested structure
      containing values that are convertible to tensors. Note that passing an
      N-dimension list of compatible values will result in a N-dimension list of
      scalar tensors rather than a single Rank-N tensors. If you need different
      behavior, convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to computation.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each replica of the computation uses
      only one core, and there is either only one replica, or the number of
      replicas is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    use_tpu: When false, the input `computation` is executed on the XLA CPU/GPU
      backends. Currently, only supports a default placement (computation is
      placed on GPU if one is available, and on CPU if not).
    maximum_shapes: A nested structure of tf.TensorShape representing the shape
      to which the respective component of each input element in each replica
      should be padded. Any unknown dimensions (e.g.
      tf.compat.v1.Dimension(None) in a tf.TensorShape or -1 in a tensor-like
      object) will be padded to the maximum size of that dimension over all
      replicas. The structure of `maximum_shapes` needs to be the same as
      `inputs[0]`.
    padding_spec: An enum specified by `tf.tpu.PaddingSpec`. This describes the
      padding policy when the `inputs` to `tf.tpu.replicate` is dynamic.
      One usage is to enable automatic bucketizing on the inputs by setting the
      value to `tpu.PaddingSpec.POWER_OF_TWO`, which can help to reduce the
      recompilation in the XLA side.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.

  Returns:
    A list of lists with the first list corresponding to the compile op and the
    second a list of output tensors, indexed by `[replica_num][output_num]`.
  Raises:
    ValueError: If all replicas do not have equal numbers of input tensors.
    ValueError: If the number of inputs per replica does not match
      the number of formal parameters to `computation`.
    ValueError: If the static `inputs` dimensions don't match with the values
      given in `maximum_shapes`.
    ValueError: If the structure of inputs per replica does not match
      the structure of `maximum_shapes`.
  """
  del name
  inputs = [[]] if inputs is None else inputs
  xla_options = xla_options or XLAOptions()

  metadata_kwargs = {}
  if device_assignment is not None:
    # Turn the Numpy array into a flattened list so we can pass it as an
    # operator attribute.
    metadata_kwargs = {
        "topology":
            device_assignment.topology.serialized(),
        "device_assignment":
            device_assignment.core_assignment.flatten().tolist()
    }
    metadata_kwargs["num_cores_per_replica"] = (
        device_assignment.num_cores_per_replica)

  # This entry is used for enabling automatic outside compilation.
  metadata_kwargs["allow_soft_placement"] = config.get_soft_device_placement()
  if config.get_soft_device_placement():
    logging.info("Automatic outside compilation is enabled. "
                 "Ops without XLA kernels will be automatically "
                 "placed on CPU.")

  if not isinstance(inputs, list):
    raise TypeError("tpu.replicate() inputs must be a list of lists/tuples, "
                    f"received {type(inputs)}")
  if any(not isinstance(inp, (list, tuple)) for inp in inputs):
    raise TypeError(
        "tpu.replicate() inputs must be a list of lists/tuples, "
        f"received types: {[type(inp) for inp in inputs]}")

  num_replicas = len(inputs)

  # No replicas? Nothing to do.
  if num_replicas == 0:
    return []

  # Checks all replicas have the same structure.
  for i in range(1, num_replicas):
    nest.assert_same_structure(inputs[0], inputs[i])

  # Explicitly read variables.
  inputs = variable_utils.convert_variables_to_tensors(inputs)
  # Flatten inputs. This structure may contain None values, which will be
  # handled later.
  flat_inputs_with_nones = [
      nest.flatten(per_replica_input, expand_composites=True)
      for per_replica_input in inputs
  ]
  # Mask parallel to one replica's inputs with True for tensors coming from
  # composites.
  is_composite = nest.flatten(nest.map_structure(
      lambda x: _flatten_and_filter_composite(x, False, True), inputs[0]))

  # Converts inputs to Tensors, replacing Nones with a placeholder 0 since
  # tpu_ops.tpu_replicated_input() can't handle non-Tensor values.
  flat_inputs = []
  for inp in flat_inputs_with_nones:
    flat_inputs.append([
        constant_op.constant(0) if x is None else ops.convert_to_tensor(x)
        for x in inp
    ])

  # Verifies that all replicas have matching numbers and types of inputs
  flat_input_types = [x.dtype for x in flat_inputs[0]]
  input_arity = len(inputs[0])
  flat_input_arity = len(flat_input_types)
  for i in range(num_replicas):
    if len(inputs[i]) != input_arity:
      raise ValueError("Replicas must have the same number of inputs. "
                       "Replica 0 had {} inputs, replica {} had {} "
                       "inputs.".format(input_arity, i, len(inputs[i])))

    types = [x.dtype for x in flat_inputs[i]]
    if types != flat_input_types:
      raise ValueError("Replicas must have matching input types. Replica 0 had "
                       "input types {}, replica {} had input types {}".format(
                           flat_input_types, i, types))

  arg_error = xla.check_function_argument_count(
      computation, input_arity, infeed_queue)
  if arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied computation cannot be called with the specified inputs. "
          f"You specified {input_arity} inputs: {[i.name for i in inputs[0]]}, "
          f"but the computation needs {arg_error}")
    else:
      raise TypeError(
          "Supplied computation cannot be called with the specified inputs. "
          f"You specified {input_arity} inputs: {[i.name for i in inputs[0]]} ",
          f"and {infeed_queue.number_of_tuple_elements} additional inputs "
          f"from infeed, but the computation needs {arg_error}")

  dynamic_shape_inputs = False
  if maximum_shapes:
    if infeed_queue:
      raise ValueError(
          "Dynamic input shapes are not supported with infeed queues")

    # Make sure maximum_shapes has the same structure as inputs.
    nest.assert_same_structure(inputs[0], maximum_shapes, check_types=False)

    # Flatten padded shapes:
    # For composite tensor components, we don't want to pad them. For each
    # entry of maximum_shapes that corresponds to a composite tensor, replace it
    # by a tuple of Nones of the same length as the number of components of the
    # composite tensor. When we flatten a second time, this makes
    # flat_maximum_shapes have the same length as flat_inputs[i]. We can then
    # avoid padding these tensors. The assumption is that they will be used by
    # outside compilation or that the components are statically shaped and will
    # be used by tpu compatible ops.
    flat_maximum_shapes = nest.flatten(
        [_flatten_and_filter_composite(x, y)
         for x, y in zip(nest.flatten(inputs[0]),
                         nest.flatten(maximum_shapes))])
    flat_maximum_shapes = [
        tensor_shape.TensorShape(s) if s is not None else None
        for s in flat_maximum_shapes
    ]
    nest.assert_same_structure(flat_inputs[0], flat_maximum_shapes,
                               check_types=False)

    unpadded_inputs = flat_inputs
    flat_inputs, padding_maps = _pad_all_input(unpadded_inputs,
                                               flat_maximum_shapes,
                                               padding_spec)
    if padding_maps:
      dynamic_shape_inputs = True
      logging.info("TPU has inputs with dynamic shapes: %s", inputs[0])

  metadata_kwargs["step_marker_location"] = getattr(
      computation, "step_marker_location", "STEP_MARK_AT_ENTRY")
  metadata_kwargs["use_spmd_for_xla_partitioning"] = \
      xla_options.use_spmd_for_xla_partitioning
  metadata_kwargs["use_shardy_partitioner"] = xla_options.use_shardy_partitioner

  graph = ops.get_default_graph()

  # Fan-in: Builds a TPUReplicatedInput node for each input.
  flat_replicated_inputs = []
  for i in range(0, len(flat_inputs[0])):
    replicas = [flat_inputs[replica][i] for replica in range(num_replicas)]
    flat_replicated_inputs.append(
        tpu_ops.tpu_replicated_input(
            replicas, name="input{}".format(i)))
  if isinstance(graph, func_graph.FuncGraph):
    # When we are in Tensorflow 2.0 function, 'graph' will be a FuncGraph
    # object. If both outside graph and this function have a TPU cluster,
    # they will have the same cluster name and it will cause problems (because
    # we lower functional ops in Tensorflow 2.0). Append function name to
    # 'cluster_name' to avoid cluster name collision.
    cluster_name = graph.unique_name("cluster_" + graph.name)
  else:
    cluster_name = graph.unique_name("cluster")
  pivot = control_flow_ops.no_op(name=cluster_name + "/pivot")
  pivot._set_attr(_PIVOT_FOR_CLUSTER,  # pylint: disable=protected-access
                  attr_value_pb2.AttrValue(s=compat.as_bytes(cluster_name)))
  context = tpu_replication.TPUReplicateContext(
      name=cluster_name, num_replicas=num_replicas, pivot=pivot)
  try:
    context.Enter()

    metadata = tpu_ops.tpu_replicate_metadata(
        num_replicas=num_replicas, use_tpu=use_tpu, **metadata_kwargs)

    with tpu_function.tpu_shard_context(
        num_replicas), ops.control_dependencies([metadata]):

      if dynamic_shape_inputs and xla_options.enable_xla_dynamic_padder:
        for padding_map in padding_maps:
          input_shape = flat_replicated_inputs[padding_map.arg_index].shape
          flat_replicated_inputs[
              padding_map.arg_index] = tf2xla.set_dynamic_dimension_size(
                  flat_replicated_inputs[padding_map.arg_index],
                  padding_map.shape_index,
                  flat_replicated_inputs[padding_map.padding_arg_index])
          flat_replicated_inputs[padding_map.arg_index].set_shape(input_shape)

      # Add identity ops so even unused inputs are "consumed" by the
      # computation. This is to avoid orphaned TPUReplicatedInput nodes.
      # TODO(phawkins): consider instead pruning unused TPUReplicatedInput
      # and eliding trivial TPUReplicatedInput/TPUReplicatedOutput pairs.
      flat_replicated_inputs = [
          array_ops.identity(x, name="replicated_input_{}".format(i))
          for i, x in enumerate(flat_replicated_inputs)
      ]
      for i, composite in zip(flat_replicated_inputs, is_composite):
        # pylint: disable=protected-access
        # Add an attribute to the identity node so that they could be removed in
        # encapsulate TPU computation pass if unused. However we don't remove
        # inputs when dynamic padding is enabled.
        # TODO(rxsang): Use other ways except argument index in padding_map so
        # outside compilation can work with dynamic padding correctly.
        if not dynamic_shape_inputs or composite:
          i.op._set_attr("_tpu_input_identity",
                         attr_value_pb2.AttrValue(b=True))
        # pylint: enable=protected-access

      # Clobber replicated placeholders with Nones.
      computation_inputs = [
          None if inp is None else replicated for replicated, inp in zip(
              flat_replicated_inputs, flat_inputs_with_nones[0])
      ]

      # Unflatten the computation inputs to match original input structure.
      computation_inputs = nest.pack_sequence_as(
          structure=inputs[0],
          flat_sequence=computation_inputs[:flat_input_arity],
          expand_composites=True)

      # If there is an infeed queue, adds the dequeued values to the
      # computation's inputs.
      if infeed_queue is not None:
        infeed_queue.set_number_of_shards(num_replicas)
        for t in infeed_queue.generate_dequeue_op():
          computation_inputs.append(t)

      # Only resource variables work inside a TPU computation, so turn on
      # resource variables for the computation.
      # TODO(phawkins): consider removing this code. It will
      # be less confusing to clients if they knowingly choose to use resource
      # variables.
      # Partitioned variables is not supported (b/112311320).
      vscope = variable_scope.get_variable_scope()
      saved_use_resource = vscope.use_resource
      saved_custom_getter = vscope.custom_getter

      def custom_getter(getter, name, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: variable-scope custom getter installed while the TPU computation is traced.
        # Inputs: the underlying getter, the variable name, and its positional and keyword
        # arguments. If a partitioner is passed it is cleared and a warning is logged, since
        # partitioned variables are unsupported on TPU. Then delegates to the previously installed
        # custom getter if any, otherwise to the plain getter. Returns the variable it produces.
        """Variables on TPU have a few restrictions."""
        partitioner = kwargs.get("partitioner", None)
        if partitioner is not None:
          kwargs["partitioner"] = None
          logging.warning(
              "Partitioned variables are not supported on TPU. Got "
              "`partitioner` that is %s for variable %s. "
              "Setting `partitioner` to `None`.", partitioner, name)
        if saved_custom_getter is None:
          return getter(name, *args, **kwargs)
        else:
          return saved_custom_getter(getter, name, *args, **kwargs)

      vscope.set_use_resource(True)
      vscope.set_custom_getter(custom_getter)

      outputs = computation(*computation_inputs)

      vscope.set_use_resource(saved_use_resource)
      vscope.set_custom_getter(saved_custom_getter)

      outputs = variable_utils.convert_variables_to_tensors(outputs)

    need_spmd_partitioning = (
        xla_options.use_spmd_for_xla_partitioning and
        device_assignment is not None and
        device_assignment.num_cores_per_replica > 1)
    outputs_is_flat = xla.is_flat(outputs)
    if outputs_is_flat:
      output_tensors, control_deps, pack_template = _postprocess_flat_outputs(
          outputs, need_spmd_partitioning)
    else:
      output_tensors, control_deps, pack_template = (
          _postprocess_non_flat_outputs(outputs, need_spmd_partitioning))

    if tensor_tracer.TensorTracer.is_enabled():
      if tf2.enabled():
        logging.warn("TF API ver >= 2.0 detected. "
                     "Tensor Tracer v1 is not enabled.")
      else:
        tt = tensor_tracer.TensorTracer()
        output_tensors = tt.trace_tpu(ops.get_default_graph(),
                                      output_tensors, control_deps,
                                      num_replicas)

    context.ExitResult(output_tensors)
  finally:
    context.report_unsupported_operations()
    context.Exit()
    host_compute_core = context.HostComputeCore()

  if host_compute_core:
    attr_value = attr_value_pb2.AttrValue()
    attr_value.list.s.extend(compat.as_bytes(x) for x in host_compute_core)
    metadata._set_attr("host_compute_core", attr_value)  # pylint: disable=protected-access

  with ops.control_dependencies([metadata]):
    if use_tpu:
      compile_status = tpu_ops.tpu_compilation_result()
      op = compile_status.op
      attr_value = attr_value_pb2.AttrValue(s=compat.as_bytes(cluster_name))
      op._set_attr(_TPU_COMPILATION_STATUS_ATTR, attr_value)  # pylint: disable=protected-access
    else:
      compile_status = control_flow_ops.no_op(name="compilation_status")

  if not output_tensors:
    # Returns a list of NoOps dependent on the replication Op, indexed by
    # [replica_num].
    return [
        compile_status,
        [
            control_flow_ops.group(control_deps, name="shard_%d" % i)
            for i in range(num_replicas)
        ]
    ]

  # Fan-out: Builds a TPUReplicatedOutput node for each output.
  replicated_outputs = [[] for i in range(num_replicas)]
  for i, t in enumerate(output_tensors):

    # None values returned by the computation can't be sent to
    # tpu_ops.tpu_replicated_output(), we handle them specially here. We can
    # avoid the placeholder 0 routine required on the inputs since outputs are
    # replicated per-tensor, not per-replica, so we can skip replication.
    if t is None:
      for replica in range(num_replicas):
        replicated_outputs[replica].append(None)
      continue

    # Fan-out: Builds a TPUReplicatedOutput node for each output.
    ys = tpu_ops.tpu_replicated_output(
        t, num_replicas, name="output{}".format(i))

    # Wraps the outputs in identity operators so the names of any possible
    # `fetch` nodes are preserved by the replication rewrite.
    with ops.control_dependencies(control_deps):
      for replica in range(num_replicas):
        replicated_outputs[replica].append(
            array_ops.identity(
                ys[replica], name="output_%d_shard_%d" % (i, replica)))

  replicated_outputs = [
      nest.pack_sequence_as(pack_template, replica_outs, expand_composites=True)
      for replica_outs in replicated_outputs
  ]

  return [compile_status, replicated_outputs]


def _postprocess_flat_outputs(
    outputs: Any,
    need_spmd_partitioning: bool
) -> Tuple[List[Optional[core_types.Tensor]], List[ops.Operation], List[Any]]:
  # [FUNCTION DOC]
  # HELPER: output handling in split_compile_and_replicate when the computation result is flat.
  # KNOWN ISSUE: the docstring says it validates non-flat outputs, but this is the flat-outputs
  # variant (the non-flat one is a separate function).
  # Inputs: outputs (result of the computation: nothing, a single value, or a flat sequence of
  # tensors and operations) and need_spmd_partitioning (when true no device is forced).
  # Nothing becomes an empty tuple; composites are flattened to components; a no-op is appended so
  # fetching any output triggers execution. Values are converted to tensors (on the first core
  # unless SPMD), raising a value error if conversion fails or if operations are not all after
  # the tensors. Tensors are wrapped in identity ops tagged as TPU output identity (on their own
  # device or the first core) so replicated input cannot bypass replication. Returns the new
  # tensor list, the list of operations (including the added no-op) and the pack template (with
  # the extra operations trimmed) for repacking outputs.
  """Validates non-flat outputs, add backs device assignments and other attrs.

  Args:
    outputs: Output from `computation` inside `tpu.rewrite`.
    need_spmd_partitioning: Whether XLA SPMD partitioning is needed.

  Returns:
    - Tensors extracted from outputs.
    - Operations extracted from outputs.
    - A pack template for use with nest.pack_sequence_as to pack the tensors.
  """
  # Following code segment is to preserve legacy behavior. Previously we only
  # supported flat outputs and thus for consistency it was nice to convert even
  # single element into a tuple. But now that we support arbitrary output
  # structure, this is no longer necessary.
  # TODO(b/121383831): Migrate all legacy use cases and delete this special
  # case.
  # If the computation returns `None`, make it an empty tuple.
  if outputs is None:
    outputs = tuple()

  # For legacy / backwards compatibility reasons we return a list for "flat"
  # output values (even if the user's flat return value was a different type or
  # even just a scalar value) so use nest.flatten to compute a flat list pack
  # template.
  pack_template = nest.flatten(outputs, expand_composites=False)

  # Even though outputs is already "flat", we flatten any composites so their
  # component tensors can be tagged and replicated. The pack_template will be
  # used by the caller to repack the composite tensors.
  outputs = nest.flatten(outputs, expand_composites=True)

  # Append `no_op` here so that fetching any return value of this function
  # will trigger TPUExecute node.
  outputs += (control_flow_ops.no_op(),)

  maybe_convert = lambda x: None if x is None else ops.convert_to_tensor(x)
  try:
    if need_spmd_partitioning:
      outputs = [
          o if isinstance(o, ops.Operation) else maybe_convert(o)
          for o in outputs
      ]
    else:
      with ops.device(core(0)):
        outputs = [
            o if isinstance(o, ops.Operation) else maybe_convert(o)
            for o in outputs
        ]
  except Exception as e:
    raise ValueError(
        "TPU function return values must all either be Operations or "
        f"convertible to Tensors. Got error: {e}")

  # Separates the returned Operations and Tensors.
  output_operations = [o for o in outputs if isinstance(o, ops.Operation)]
  output_tensors = [o for o in outputs if not isinstance(o, ops.Operation)]

  if outputs != output_tensors + output_operations:
    raise ValueError(
        "TPU functions must return zero-or more Tensor values followed by "
        "zero or more Operations.")

  # Trim operations off the end of the pack template. output_operations has 1
  # extra element due to the no-op that is added.
  if len(output_operations) > 1:
    pack_template = pack_template[:1 - len(output_operations)]

  # Wraps outputs in Identity ops. Otherwise a replicated input copied
  # straight to an output would bypass the replicate(). This would be bad
  # because the TPUReplicatedInput/TPUReplicatedOutput operator would not
  # be rewritten away, leading to a runtime error.
  # TODO(phawkins): extend the rewrite to elide these nodes instead.
  new_output_tensors = []
  for t in output_tensors:
    if t is None:
      new_output_tensors.append(None)
    elif need_spmd_partitioning:
      o = array_ops.identity(t)
      # pylint: disable=protected-access
      o.op._set_attr("_tpu_output_identity", attr_value_pb2.AttrValue(b=True))
      # pylint: enable=protected-access
      new_output_tensors.append(o)
    else:
      with ops.device(t.device if t.device else core(0)):
        o = array_ops.identity(t)
        # pylint: disable=protected-access
        o.op._set_attr("_tpu_output_identity", attr_value_pb2.AttrValue(b=True))
        # pylint: enable=protected-access
        new_output_tensors.append(o)
  return new_output_tensors, output_operations, pack_template


def _postprocess_non_flat_outputs(
    outputs: Any,
    need_spmd_partitioning: bool
) -> Tuple[List[Optional[core_types.Tensor]], List[ops.Operation], List[Any]]:
  # [FUNCTION DOC]
  # HELPER: output handling in split_compile_and_replicate when the computation result is nested.
  # Inputs: outputs (arbitrary nested structure of tensors, convertible values, composites or
  # nothing values) and need_spmd_partitioning (when true no device is forced on the identities).
  # Flattens the structure (expanding composites); nothing values stay nothing; every other value
  # is converted to a tensor and wrapped in identity ops tagged as TPU output identity, placed on
  # its own device or the first core unless SPMD is used. Raises a value error if any element is
  # an operation (not allowed in nested outputs) or cannot be converted to a tensor.
  # Returns the flat tensor list, an empty operations list, and the original structure as the
  # pack template.
  """Validates non-flat outputs, add backs device assignments and other attrs.

  Args:
    outputs: Output from `computation` inside `tpu.rewrite`.
    need_spmd_partitioning: Whether XLA SPMD partitioning is needed.

  Returns:
    - Tensors extracted from outputs.
    - An empty Operations list because Operations are not allowed in non-flat
      outputs.
    - A pack template for use with nest.pack_sequence_as to pack the tensors.
  """

  # Flatten output items.
  flat_outputs = nest.flatten(outputs, expand_composites=True)

  # Convert all non-None non-Operation outputs to Tensors.
  for i, o in enumerate(flat_outputs):
    if o is None:
      flat_outputs[i] = None
      continue

    if isinstance(o, ops.Operation):
      raise ValueError(
          "tpu.rewrite does not support Operation as return value in non-flat "
          "output structure. You can set returned Operations as control "
          "dependencies of returned Tensors so Operations are triggered when "
          f'Tensors are evaluated. Operation found: "{o.name}"')

    try:
      o = ops.convert_to_tensor(o)
    except Exception as e:
      raise ValueError(
          "TPU function return values must all either be Operations or "
          f'convertible to Tensors. Got error: "{e}"')

    # Wraps outputs in Identity ops. Otherwise a replicated input copied
    # straight to an output would bypass the replicate(). This would be bad
    # because the TPUReplicatedInput/TPUReplicatedOutput operator would not
    # be rewritten away, leading to a runtime error.
    # TODO(phawkins): extend the rewrite to elide these nodes instead.
    if need_spmd_partitioning:
      o = array_ops.identity(o)
      # pylint: disable=protected-access
      o.op._set_attr("_tpu_output_identity", attr_value_pb2.AttrValue(b=True))
      # pylint: enable=protected-access
      flat_outputs[i] = array_ops.identity(o)
    else:
      with ops.device(o.device if o.device else core(0)):
        o = array_ops.identity(o)
        # pylint: disable=protected-access
        o.op._set_attr("_tpu_output_identity", attr_value_pb2.AttrValue(b=True))
        # pylint: enable=protected-access
        flat_outputs[i] = array_ops.identity(o)

  # All flat_outputs are Tensors, and no Operations.
  return flat_outputs, [], outputs


def split_compile_and_shard(
    computation: Callable[..., Any],
    inputs: Optional[List[List[Optional[core_types.Tensor]]]] = None,
    num_shards: int = 1,
    input_shard_axes: Optional[List[int]] = None,
    outputs_from_all_shards: Union[bool, List[bool]] = True,
    output_shard_axes: Optional[List[int]] = None,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[device_assignment_lib.DeviceAssignment] = None,
    name: Optional[Text] = None,
    xla_options: Optional[XLAOptions] = None,
    ) -> Tuple[ops.Operation, List[core_types.Tensor]]:
  # [FUNCTION DOC]
  # Lower level sharding builder returning the compile op separately from the outputs.
  # Inputs: computation (callable applied to each shard), inputs (list of tensors or nothing),
  # num_shards (positive integer, default 1), input_shard_axes (per input split axis; nothing
  # means axis 0 for all), outputs_from_all_shards (a boolean or per-output booleans; true
  # concatenates outputs across shards, false takes the first shard's value), output_shard_axes
  # (per-output concatenation axis; nothing means 0), infeed_queue, device_assignment, deprecated
  # name, xla_options. Each input is converted to a tensor and split into equal pieces along its
  # axis; the pieces are arranged per shard and passed to split_compile_and_replicate. When the
  # computation produced no tensors it returns the compile op and a one element list with the
  # first group no-op. Otherwise outputs are concatenated along their axis (scalars are stacked)
  # or taken from shard zero.
  # Raises a value error for non-positive shard count or length mismatches between inputs and
  # input axes, outputs and output axes, or outputs and the per-output flags; a type error if
  # inputs is not a list. Returns the pair of compile op and list of output tensors.
  """Shards `computation` for parallel execution.

  `inputs` must be a list of Tensors or None (equivalent to an empty list), each
  of which has a corresponding split axis (from `input_shard_axes`). Each input
  is split into `num_shards` pieces along the corresponding axis, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  If `outputs_from_all_shards` is true, the outputs from all shards of
  `computation` are concatenated back together along their `output_shard_axes`.
  Otherwise, each output is taken from an arbitrary shard.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). Each
      input tensor has a corresponding shard axes, given by `input_shard_axes`,
      which must have size divisible by `num_shards`.
    num_shards: The number of shards.
    input_shard_axes: A list of dimensions along which to shard `inputs`, or
      `None`. `None` means "shard all inputs along dimension 0". If not `None`,
      there must be one dimension per input.
    outputs_from_all_shards: Boolean or list of boolean. For each output, if
      `True`, outputs from all shards are concatenated along the corresponding
      `output_shard_axes` entry. Otherwise, each output is taken
      from an arbitrary shard. If the argument is a boolean, the argument's
      value is used for each output.
    output_shard_axes: A list of dimensions along which to concatenate the
      outputs of `computation`, or `None`. `None` means "concatenate all outputs
      along dimension 0". If not `None`, there must be one dimension per output.
      Ignored if `outputs_from_all_shards` is False.
    infeed_queue: If not `None`, the `InfeedQueue` to use to augment the inputs
      of `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.
  Returns:
    A tuple of (compile op, [output tensors]).
  Raises:
    ValueError: If num_shards <= 0
    ValueError: If len(input_shard_axes) != len(inputs)
    ValueError: If len(output_shard_axes) != len(outputs from `computation`)
  """
  # TODO(phawkins): consider adding support for broadcasting Tensors passed as
  # inputs.

  if num_shards <= 0:
    raise ValueError(
        f"num_shards must be a positive integer. Received {num_shards}")

  inputs = [] if inputs is None else inputs
  if not isinstance(inputs, list):
    raise TypeError("tpu.shard()'s inputs must be a list of Tensors or None. "
                    f"Received {type(inputs)}")

  # Converts inputs to Tensors.
  inputs = [ops.convert_to_tensor(x) for x in inputs]

  if input_shard_axes is None:
    input_shard_axes = [0] * len(inputs)
  if len(inputs) != len(input_shard_axes):
    raise ValueError("Length of input_shard_axes must be equal to the number "
                     f"of inputs. Received {len(inputs)} inputs and "
                     f"{len(input_shard_axes)} input_shard_axes.")

  if inputs:
    # Splits the `inputs` along the corresponding `input_shard_axes`, giving
    # lists with layout [input][shard]
    split_inputs = [
        array_ops.split(x, num_shards, axis=axis)
        for (axis, x) in zip(input_shard_axes, inputs)]

    # Transposes the input lists to have layout [shard][input]
    transposed_inputs = [list(i) for i in zip(*split_inputs)]
  else:
    transposed_inputs = [[]] * num_shards

  compile_op, outputs = split_compile_and_replicate(
      computation,
      transposed_inputs,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name,
      xla_options=xla_options)

  # There must be at least one shard since num_shards > 0.
  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  if isinstance(outputs[0], ops.Operation):
    # pylint: enable=indexing-exception
    # There were no outputs from the computation and replicate returned a list
    # of NoOps with control dependencies on the computation. Return the first
    # one so it can be used as a control dependency or fetch node.
    # TODO(b/36647078) remove disable when pylint bug is fixed.
    # pylint: disable=indexing-exception
    return compile_op, [outputs[0]]
    # pylint: enable=indexing-exception

  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  num_outputs = len(outputs[0])
  # pylint: enable=indexing-exception

  if output_shard_axes is None:
    output_shard_axes = [0] * num_outputs
  if num_outputs != len(output_shard_axes):
    raise ValueError("Length of output_shard_axes must be equal to the number "
                     f"of outputs. Received {num_outputs} outputs "
                     f"and {len(output_shard_axes)} output_shard_axes.")

  if isinstance(outputs_from_all_shards, bool):
    outputs_from_all_shards = [outputs_from_all_shards] * num_outputs

  if num_outputs != len(outputs_from_all_shards):
    raise ValueError(
        "Length of outputs_from_all_shards must be equal to the number of "
        f"outputs. Received {num_outputs} outputs  and "
        f"{len(outputs_from_all_shards)} outputs_from_all_shards.")

  results = []
  for (axis, all_shards, x) in zip(output_shard_axes, outputs_from_all_shards,
                                   zip(*outputs)):
    if all_shards:
      # Concatenate all of the outputs together (use stack for scalars).
      shape = x[0].shape
      is_scalar = shape is not None and (shape.ndims == 0)
      results.append((array_ops_stack.stack(list(x)) if is_scalar
                      else array_ops.concat(list(x), axis=axis)))
    else:
      # TODO(phawkins): use a smarter policy, e.g., round-robin across shards.
      results.append(x[0])

  return compile_op, results


@tf_export(v1=["tpu.shard"])
@traceback_utils.filter_traceback
def shard(
    computation: Callable[..., Any],
    inputs: Optional[List[core_types.Tensor]] = None,
    num_shards: int = 1,
    input_shard_axes: Optional[List[int]] = None,
    outputs_from_all_shards: Union[bool, List[bool]] = True,
    output_shard_axes: Optional[List[int]] = None,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[device_assignment_lib.DeviceAssignment] = None,
    name: Optional[Text] = None,
    xla_options: Optional[XLAOptions] = None) -> List[core_types.Tensor]:
  # [FUNCTION DOC]
  # Public sharding wrapper. Forwards all arguments to the split compile-and-shard builder and
  # returns only its second element, the list of output tensors (concatenated across shards where
  # requested). Errors (invalid shard counts, axis length mismatches, non-list inputs) come from
  # the builder. Traceback filtering is applied by decorator.
  """Shards `computation` for parallel execution.

  `inputs` must be a list of Tensors or None (equivalent to an empty list), each
  of which has a corresponding split axis (from `input_shard_axes`). Each input
  is split into `num_shards` pieces along the corresponding axis, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  TODO(phawkins): consider adding support for broadcasting Tensors passed
  as inputs.

  If `outputs_from_all_shards` is true, the outputs from all shards of
  `computation` are concatenated back together along their `output_shard_axes`.
  Otherwise, each output is taken from an arbitrary shard.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). Each
      input tensor has a corresponding shard axes, given by `input_shard_axes`,
      which must have size divisible by `num_shards`.
    num_shards: The number of shards.
    input_shard_axes: A list of dimensions along which to shard `inputs`, or
      `None`. `None` means "shard all inputs along dimension 0". If not `None`,
      there must be one dimension per input.
    outputs_from_all_shards: Boolean or list of boolean. For each output, if
      `True`, outputs from all shards are concatenated along the corresponding
      `output_shard_axes` entry. Otherwise, each output is taken
      from an arbitrary shard. If the argument is a boolean, the argument's
      value is used for each output.
    output_shard_axes: A list of dimensions along which to concatenate the
      outputs of `computation`, or `None`. `None` means "concatenate all outputs
      along dimension 0". If not `None`, there must be one dimension per output.
      Ignored if `outputs_from_all_shards` is False.
    infeed_queue: If not `None`, the `InfeedQueue` to use to augment the inputs
      of `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: If num_shards <= 0
    ValueError: If len(input_shard_axes) != len(inputs)
    ValueError: If len(output_shard_axes) != len(outputs from `computation`)
  """
  return split_compile_and_shard(
      computation,
      inputs=inputs,
      num_shards=num_shards,
      input_shard_axes=input_shard_axes,
      outputs_from_all_shards=outputs_from_all_shards,
      output_shard_axes=output_shard_axes,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name,
      xla_options=xla_options)[1]


@tf_export(v1=["tpu.batch_parallel"])
@traceback_utils.filter_traceback
def batch_parallel(
    computation: Callable[..., Any],
    inputs: Optional[List[List[Optional[core_types.Tensor]]]] = None,
    num_shards: int = 1,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[device_assignment_lib.DeviceAssignment] = None,
    name: Optional[Text] = None,
    xla_options: Optional[XLAOptions] = None):
  # [FUNCTION DOC]
  # Convenience wrapper around shard that splits every input along dimension zero and
  # concatenates every output along dimension zero. Inputs: computation, inputs (list of tensors
  # or nothing), num_shards (default 1), infeed_queue, device_assignment, deprecated name and
  # xla_options. Returns the list of output tensors from shard. Raises a value error if the number
  # of shards is not positive (raised downstream).
  """Shards `computation` along the batch dimension for parallel execution.

  Convenience wrapper around shard().

  `inputs` must be a list of Tensors or None (equivalent to an empty list).
  Each input is split into `num_shards` pieces along the 0-th dimension, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  The outputs from all shards are concatenated back together along their 0-th
  dimension.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). The
      0-th dimension of each Tensor must have size divisible by `num_shards`.
    num_shards: The number of shards.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: If `num_shards <= 0`
  """
  return shard(
      computation,
      inputs,
      num_shards=num_shards,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name,
      xla_options=xla_options)


@tf_export(v1=["tpu.rewrite"])
@traceback_utils.filter_traceback
def rewrite(
    computation: Callable[..., Any],
    inputs: Optional[List[List[Optional[core_types.Tensor]]]] = None,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[device_assignment_lib.DeviceAssignment] = None,
    name: Optional[Text] = None,
    xla_options: Optional[XLAOptions] = None) -> Any:
  # [FUNCTION DOC]
  # Public single-replica entry point. Wraps the given inputs (list of values, or nothing) as the
  # only replica and calls replicate with the computation, infeed queue, device assignment,
  # deprecated name and XLA options, then returns the first replica's outputs in the structure the
  # computation returned (with the special cases of nothing, single value or operation-only
  # outputs described in the docstring). Errors come from replicate.
  """Rewrites `computation` for execution on a TPU system.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors.

      `computation` may return a list of operations and tensors. Tensors must
      come before operations in the returned list.  The return value of
      `rewrite` is a list of tensors corresponding to the tensors from the
      output of `computation`.

      All `Operation`s constructed during `computation` will be executed when
      evaluating any of the returned output tensors, not just the ones returned.
    inputs: A list of input tensors or `None` (equivalent to an empty list).
      Each input can be a nested structure containing values that are
      convertible to tensors. Note that passing an N-dimension list of
      compatible values will result in a N-dimension list of scalar tensors
      rather than a single Rank-N tensors. If you need different behavior,
      convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: if not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. May be omitted for a single-core computation, in which
      case the core attached to task 0, TPU device 0 is used.
    name: (Deprecated) Does nothing.
    xla_options: An instance of `tpu.XLAOptions` which indicates the options
      passed to XLA compiler. Use `None` for default options.
  Returns:
    Same data structure as if computation(*inputs) is called directly with some
    exceptions for correctness. Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
      TODO(b/121383831): Investigate into removing these special cases.
  """
  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  return replicate(
      computation,
      None if inputs is None else [inputs],
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name,
      xla_options=xla_options)[0]
  # pylint: enable=indexing-exception

  # Operations that indicate some error in the user's inference graph.


_DENYLISTED_INFERENCE_OPS = set([
    "ReadVariableOp",
    "AssignVariableOp",
    "AssignAddVariableOp",
    "AssignSubVariableOp",
    "VarHandleOp",
    "Variable",
    "VariableV2",
])


def under_tpu_inference_context() -> bool:
  # [FUNCTION DOC]
  # Reports whether graph construction is currently inside a TPU inference context. Walks the
  # control-flow contexts of the default graph (and their outer contexts), then moves to the outer
  # graph of function graphs, until an inference context is found. Returns true if one is found,
  # false when the outermost graph is reached. Takes no inputs and has no side effects.
  """Check if it is currently under `_TPUInferenceContext`."""
  graph = ops.get_default_graph()
  while graph:
    context = graph._get_control_flow_context()  # pylint: disable=protected-access
    while context:
      if isinstance(context, _TPUInferenceContext):
        return True
      context = context.outer_context
    if isinstance(graph, function._FuncGraph):  # pylint: disable=protected-access
      graph = graph._outer_graph  # pylint: disable=protected-access
    elif isinstance(graph, func_graph.FuncGraph):
      graph = graph.outer_graph
    else:
      return False
  return False


class _TPUInferenceContext(control_flow_ops.XLAControlFlowContext):
  # [CLASS DOC]
  # HELPER: control-flow context used by rewrite_for_inference. Marks that ops are being built for
  # a TPU inference computation and, if enabled, rejects variable-related ops that cannot run
  # there. Holds a name and a flag for op checking; forwards ops and values to the outer context.
  """A `ControlFlowContext` for nodes inside a TPU inference computation.

  The primary role of `_TPUInferenceContext` is to indicate the mode of
  operation and possibly sanity check operators inside a
  tpu.rewrite_for_inference() computation.
  """

  def __init__(self, name: Text, check_ops: bool = True):
    # [FUNCTION DOC]
    # Constructor. Inputs: name (context name) and check_ops (default true; whether to reject
    # unsupported variable ops). Initializes the base context and stores both. Returns nothing.
    super(_TPUInferenceContext, self).__init__()
    self._name = name
    self._check_ops = check_ops

  def AddOp(self, op):
    # [FUNCTION DOC]
    # Called when an op is created in this context. Input: the op. Delegates to the internal check
    # (which may raise) and registers the op with the outer context. Returns nothing.
    self._AddOpInternal(op)

  def _AddOpInternal(self, op):
    # [FUNCTION DOC]
    # HELPER: shared logic for AddOp and AddInnerOp. Input: the op. If checking is enabled and the
    # op type is on the deny list of variable ops (read, assign, handle creation, legacy variable
    # types), raises a not-implemented error explaining it is unsupported on TPU for inference.
    # Otherwise forwards the op to the outer context, if any. Returns nothing.
    # pylint: disable=protected-access
    if self._check_ops and op.type in _DENYLISTED_INFERENCE_OPS:
      raise NotImplementedError(
          f"Operation of type {op.type} ({op.name}) is not supported on the "
          "TPU for inference. Execution will fail if this op is used in the "
          "graph. Make sure your variables are using variable_scope.")
    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def AddValue(self, val):
    # [FUNCTION DOC]
    # Called when an external value is used inside the context. Input: the value. Returns the
    # value unchanged if there is no outer context, otherwise whatever the outer context's
    # AddValue returns for it.
    result = val
    if self._outer_context:
      result = self._outer_context.AddValue(val)
    return result

  def AddInnerOp(self, op):
    # [FUNCTION DOC]
    # Called when an op is created in a nested context. Input: the op. Same handling as AddOp:
    # runs the deny-list check (may raise a not-implemented error) and notifies the outer context.
    # Returns nothing.
    self._AddOpInternal(op)

  @property
  def grad_state(self):
    # [FUNCTION DOC]
    # Property required by the control-flow context interface. Always returns nothing because this
    # context does not take part in gradient loop state.
    return None


def validate_inference_rewrite_for_variables(graph: ops.Graph):
  # [FUNCTION DOC]
  # Sanity check to run right after rewrite_for_inference. Input: the graph to inspect. Scans all
  # its operations for a guarantee-constant op; raises a runtime error advising use of the
  # variable-getter API if none exists. Returns nothing; does not modify the graph.
  """Validates whether rewrite_for_inference() 'worked' for variables.

     The rewrite_for_inference() method is supposed to append GuaranteeConstOps
     after ReadVariableOps, but this mechanism works only if you are using
     tf.compat.v1.get_variable() to create and access variables in your tpu
     computation. This validation method can be called immediately after calling
     tpu.rewrite_for_inference() to check whether GuaranteeConstOps where added
     to the graph.

     Typical usages:
       tpu.validate_inference_rewrite_for_variables(
           tf.compat.v1.get_default_graph())

       tpu.validate_inference_rewrite_for_variables(sess.graph)

  Args:
    graph: The graph which needs to be validated.
  Raises:
    RuntimeError: if validation failed.
  """
  if not any(x.type == "GuaranteeConst" for x in graph.get_operations()):
    raise RuntimeError(
        "No GuaranteeConst ops found in the graph after running "
        "tpu.rewrite_for_inference(...). Please check that you are using "
        "tf.get_variable() to create and access variables in your tpu "
        "computation.")


def rewrite_for_inference(
    computation: Callable[..., Any],
    inputs: Optional[List[core_types.Tensor]] = None,
    infeed_queue: Optional[tpu_feed.InfeedQueue] = None,
    device_assignment: Optional[device_assignment_lib.DeviceAssignment] = None,
    name: Optional[Text] = None) -> List[core_types.Tensor]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the wrapped computation restores the variable scope's custom getter and caching
  # device only on success; if the computation raises, they are not restored (only the context
  # exit is in the cleanup path). It also passes name through to rewrite, which ignores it.
  # Rewrites a computation for TPU inference. Inputs: computation, inputs (list of tensors or
  # nothing), infeed_queue, device_assignment and name. Wraps the computation so it runs inside an
  # inference context with a variable getter that adds a guarantee-constant op after each variable
  # read (with no control dependencies) and a caching device that follows the op device, then
  # calls rewrite. Returns the list of output tensors.
  """Rewrites `computation` for inference on a TPU system.

     Other than 'rewriting' the computation to run on a TPU, if using variables
     in your computation, it moves the ReadVariableOps outside the TPU
     computation, and adds GuaranteeConst ops just after the ReadVariableOps.
     This mechanism works only if you are using tf.compat.v1.get_variable() to
     create and access variables in your tpu computation. You can validate
     whether this worked, by calling validate_inference_rewrite_for_variables()
     method immediately after this method to check whether GuaranteeConstOps
     where added to the graph.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors. If the function returns m outputs, rewrite will return a list of
      m tensors.
    inputs: A list of input tensors or `None` (equivalent to an empty list).
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: if not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. May be omitted for a single-core computation, in which
      case the core attached to task 0, TPU device 0 is used.
    name: The name of the operator.
  Returns:
    A list of output tensors.
  """

  def guarantee_const_getter(getter, name, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: custom variable getter installed during inference rewriting. Calls the underlying
    # getter with the given name and arguments, outside any control dependencies, and wraps the
    # resulting value in a guarantee-constant op named after the variable. Returns that tensor.
    with ops.control_dependencies(None):
      return array_ops.guarantee_const(
          getter(name, *args, **kwargs), name=name + "/GuaranteeConst")

  def wrapped_computation(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: computation passed to rewrite. Enters a new inference context, temporarily installs
    # the guarantee-constant getter and a device-following caching device on the current variable
    # scope, calls the user computation with the given arguments, restores the scope settings and
    # exits the context (always). Returns the computation's result.
    """Execute computation under `_TPUInferenceContext`."""
    context = _TPUInferenceContext(
        name=ops.get_default_graph().unique_name("rewrite_for_inference"))
    try:
      context.Enter()

      vscope = variable_scope.get_variable_scope()
      prev_custom_getter = vscope.custom_getter
      prev_caching_device = vscope.caching_device
      vscope.set_custom_getter(guarantee_const_getter)
      vscope.set_caching_device(lambda op: op.device)

      result = computation(*args, **kwargs)

      vscope.set_custom_getter(prev_custom_getter)
      vscope.set_caching_device(prev_caching_device)
    finally:
      context.Exit()
    return result

  # pylint: disable=undefined-variable
  return rewrite(
      wrapped_computation,
      inputs=inputs,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name)
  # pylint: enable=undefined-variable


def prune_unconnected_ops_from_xla(prune_graph: ops.Graph):
  # [FUNCTION DOC]
  # Removes the TPU replicate attribute from leftover input-less ops (placeholders and variable
  # handles) whose outputs have no consumers, so that they do not break XLA compilation. Input:
  # the graph; it and its function graphs are scanned. Logs each pruned op and mutates the ops'
  # attributes. Returns nothing.
  """Prunes unconnected ops as listed in _UNCONNECTED_OPS_TO_PRUNE.

  Args:
    prune_graph: A tensorflow graph from which we wish to prune unconnected ops
      as listed in _UNCONNECTED_OPS_TO_PRUNE.  In general, these ops should have
      no inputs and no consumers. These can often be left behind due to graph
      construction rewiring (for instance TF-Hub). While they never execute,
      they will cause XLA compile to fail so we strip them from XLA compile by
      removing the tpu_replicate attribute.
  """
  # Scan over the top level graph and all function graphs.
  for graph in [prune_graph] + [
      f for f in prune_graph._functions.values()  # pylint: disable=protected-access
  ]:
    if not isinstance(graph, ops.Graph):
      continue
    for op in graph.get_operations():
      if op.type not in _UNCONNECTED_OPS_TO_PRUNE:
        continue
      outputs_consumed = False
      for output in op.outputs:
        if output.consumers():
          outputs_consumed = True
          break
      if not outputs_consumed:
        logging.info(
            "Pruning OP %s of type %s from XLA Compile due to "
            "it being disconnected.", op.name, op.type)
        op._clear_attr(tpu_replication._TPU_REPLICATE_ATTR)  # pylint: disable=protected-access
