# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# HELPER: internal implementation module behind the Dataset map transformation in dataset_ops.
# Provides the version 2 and version 1 entry functions (plus a legacy-function variant) that
# choose between a sequential map dataset and a parallel map dataset, and the two dataset
# classes wrapping the map and parallel-map ops. Each class traces the user function into a
# graph function through the structured function wrapper. No import-time side effects.
"""The implementation of `tf.data.Dataset.map`."""

import warnings

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _map_v2(
    input_dataset,  # pylint: disable=unused-private-name
    map_func,
    num_parallel_calls=None,
    deterministic=None,
    synchronous=None,
    use_unbounded_threadpool=None,
    name=None,
):
  # [FUNCTION DOC]
  # HELPER: backs the version 2 Dataset map method.
  # Inputs: input dataset, map function, num_parallel_calls (optional, nothing means sequential),
  # deterministic (optional boolean), synchronous (optional boolean forcing synchronous
  # execution), use_unbounded_threadpool (optional boolean) and name (optional).
  # If parallelism is unset or debug mode is on, returns a sequential map dataset that preserves
  # cardinality; a warning is emitted if deterministic was given outside debug mode since it has
  # no effect. Otherwise returns a parallel map dataset; raises a value error if synchronous is
  # truthy together with num_parallel_calls.
  """See `Dataset.map()` for details."""
  if num_parallel_calls is None or debug_mode.DEBUG_MODE:
    if deterministic is not None and not debug_mode.DEBUG_MODE:
      warnings.warn(
          "The `deterministic` argument has no effect unless the "
          "`num_parallel_calls` argument is specified."
      )
    return _MapDataset(
        input_dataset,
        map_func,
        preserve_cardinality=True,
        force_synchronous=False if synchronous is None else synchronous,
        name=name,
    )
  else:
    if synchronous:
      raise ValueError(
          "`synchronous` is not supported with `num_parallel_calls`, but"
          " `num_parallel_calls` was set to ",
          num_parallel_calls,
      )
    return _ParallelMapDataset(
        input_dataset,
        map_func,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
        preserve_cardinality=True,
        use_unbounded_threadpool=use_unbounded_threadpool,
        name=name)


def _map_v1(
    input_dataset,  # pylint: disable=unused-private-name
    map_func,
    num_parallel_calls=None,
    deterministic=None,
    synchronous=None,
    use_unbounded_threadpool=None,  # pylint: disable=unused-argument
):
  # [FUNCTION DOC]
  # HELPER: backs the version 1 Dataset map method.
  # KNOWN ISSUE: the use_unbounded_threadpool argument is accepted but ignored; the parallel
  # path always passes false. Also no warning is given for deterministic without parallelism.
  # Same selection logic as the version 2 function but wraps the result in a version 1 dataset
  # adapter, does not preserve cardinality, and takes no name. Raises a value error if
  # synchronous is truthy together with num_parallel_calls.
  """See `Dataset.map()` for details."""
  if num_parallel_calls is None or debug_mode.DEBUG_MODE:
    return dataset_ops.DatasetV1Adapter(
        _MapDataset(
            input_dataset,
            map_func,
            preserve_cardinality=False,
            force_synchronous=False if synchronous is None else synchronous,
        )
    )
  else:
    if synchronous:
      raise ValueError(
          "`synchronous` is not supported with `num_parallel_calls`, but"
          " `num_parallel_calls` was set to ",
          num_parallel_calls,
      )
    return dataset_ops.DatasetV1Adapter(
        _ParallelMapDataset(
            input_dataset,
            map_func,
            num_parallel_calls,
            deterministic,
            preserve_cardinality=False,
            use_unbounded_threadpool=False))


def _map_v1_with_legacy_function(  # pylint: disable=unused-private-name
    input_dataset,
    map_func,
    num_parallel_calls=None,
    deterministic=None,
    synchronous=False,
):
  # [FUNCTION DOC]
  # HELPER: backs the version 1 map method variant that accepts legacy (non-tracing) functions.
  # Inputs: input dataset, map function, optional num_parallel_calls, deterministic and
  # synchronous flag (default false). Without parallelism, warns if deterministic was given and
  # returns an adapted sequential map dataset using the legacy function mode; with parallelism,
  # raises a value error if synchronous is truthy, else returns an adapted parallel map dataset
  # in legacy mode. Cardinality is not preserved; the unbounded threadpool is disabled.
  """See `Dataset.map()` for details."""
  if num_parallel_calls is None:
    if deterministic is not None:
      warnings.warn("The `deterministic` argument has no effect unless the "
                    "`num_parallel_calls` argument is specified.")
    return dataset_ops.DatasetV1Adapter(
        _MapDataset(
            input_dataset,
            map_func,
            force_synchronous=synchronous,
            preserve_cardinality=False,
            use_legacy_function=True,
        )
    )
  else:
    if synchronous:
      raise ValueError(
          "`synchronous` is not supported with `num_parallel_calls`, but"
          " `num_parallel_calls` was set to ",
          num_parallel_calls,
      )
    return dataset_ops.DatasetV1Adapter(
        _ParallelMapDataset(
            input_dataset,
            map_func,
            num_parallel_calls,
            deterministic,
            preserve_cardinality=False,
            use_legacy_function=True,
            use_unbounded_threadpool=False))


class _MapDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: dataset class used by the map transformation for sequential mapping.
  # Unary dataset that applies a traced function to each input element; its element spec is the
  # function's output structure and it exposes the function for tracking.
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(
      self,
      input_dataset,
      map_func,
      force_synchronous=False,
      use_inter_op_parallelism=True,
      preserve_cardinality=True,
      use_legacy_function=False,
      name=None,
  ):
    # [FUNCTION DOC]
    # Inputs: input dataset, map function, force_synchronous (default false),
    # use_inter_op_parallelism (default true), preserve_cardinality (default true),
    # use_legacy_function (default false) and optional name. Traces the function against the
    # input dataset's structure, stores the options, creates the map dataset op with the
    # function and its captured inputs, and initializes the base unary dataset.
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._preserve_cardinality = preserve_cardinality
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    self._force_synchronous = force_synchronous
    self._name = name
    variant_tensor = gen_dataset_ops.map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        force_synchronous=self._force_synchronous,
        **self._common_args
    )
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # Returns a list holding the traced map function wrapper.
    return [self._map_func]

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the output structure of the traced map function.
    return self._map_func.output_structure

  def _transformation_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this transformation, used when tracing the function.
    return "Dataset.map()"


class _ParallelMapDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: dataset class used by the map transformation for parallel mapping.
  # Like the sequential map dataset but runs the function with a given degree of parallelism and
  # a determinism setting (default, true or false), optionally on an unbounded threadpool.
  """A `Dataset` that maps a function over elements in its input in parallel."""

  def __init__(self,
               input_dataset,
               map_func,
               num_parallel_calls,
               deterministic,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False,
               use_unbounded_threadpool=False,
               name=None):
    # [FUNCTION DOC]
    # Inputs: input dataset, map function, num_parallel_calls (integer or tensor, converted to a
    # 64-bit integer tensor), deterministic (nothing, true or false, stored as a default, true
    # or false string), use_inter_op_parallelism (default true), preserve_cardinality (default
    # false), use_legacy_function (default false), use_unbounded_threadpool (default false) and
    # optional name. Traces the function, creates the parallel map dataset op and initializes
    # the base unary dataset.
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    if deterministic is None:
      self._deterministic = "default"
    elif deterministic:
      self._deterministic = "true"
    else:
      self._deterministic = "false"
    self._preserve_cardinality = preserve_cardinality
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    self._use_unbounded_threadpool = use_unbounded_threadpool
    self._name = name
    variant_tensor = gen_dataset_ops.parallel_map_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        num_parallel_calls=self._num_parallel_calls,
        deterministic=self._deterministic,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        use_unbounded_threadpool=self._use_unbounded_threadpool,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # Returns a list holding the traced map function wrapper.
    return [self._map_func]

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the output structure of the traced map function.
    return self._map_func.output_structure

  def _transformation_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this transformation, used when tracing the function.
    return "Dataset.map()"
