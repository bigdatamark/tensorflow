# [MODULE DOC]
# Defines DatasetCreator, a wrapper around a dataset-producing callable that Keras fit accepts as
# input data for distributed training (e.g. parameter server). Exposed publicly as an
# experimental utility. No import-time side effects.
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
# pylint: disable=g-classes-have-attributes
"""Input dataset creator for `model.fit`."""

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.types import data as data_types


class DatasetCreator(object):
  # [CLASS DOC]
  # Wrapper holding a dataset-producing callable (dataset_fn) and optional distribution input
  # options. Passed to Keras fit as the input; fit invokes the instance with an input context so
  # each worker builds its own dataset. State: dataset_fn and input_options attributes. Calling an
  # instance forwards to dataset_fn and validates the result is a dataset.
  """Object that returns a `tf.data.Dataset` upon invoking.

  `tf.keras.utils.experimental.DatasetCreator` is designated as a supported type
  for `x`, or the input, in `tf.keras.Model.fit`. Pass an instance of this class
  to `fit` when using a callable (with a `input_context` argument) that returns
  a `tf.data.Dataset`.

  ```python
  model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
  model.compile(tf.keras.optimizers.SGD(), loss="mse")

  def dataset_fn(input_context):
    global_batch_size = 64
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat()
    dataset = dataset.shard(
        input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset

  input_options = tf.distribute.InputOptions(
      experimental_fetch_to_device=True,
      experimental_per_replica_buffer_size=2)
  model.fit(tf.keras.utils.experimental.DatasetCreator(
      dataset_fn, input_options=input_options), epochs=10, steps_per_epoch=10)
  ```

  `Model.fit` usage with `DatasetCreator` is intended to work across all
  `tf.distribute.Strategy`s, as long as `Strategy.scope` is used at model
  creation:

  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver)
  with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
  model.compile(tf.keras.optimizers.SGD(), loss="mse")
  ...
  ```

  Note: When using `DatasetCreator`, `steps_per_epoch` argument in `Model.fit`
  must be provided as the cardinality of such input cannot be inferred.

  Args:
    dataset_fn: A callable that takes a single argument of type
      `tf.distribute.InputContext`, which is used for batch size calculation and
      cross-worker input pipeline sharding (if neither is needed, the
      `InputContext` parameter can be ignored in the `dataset_fn`), and returns
      a `tf.data.Dataset`.
    input_options: Optional `tf.distribute.InputOptions`, used for specific
      options when used with distribution, for example, whether to prefetch
      dataset elements to accelerator device memory or host device memory, and
      prefetch buffer size in the replica device memory. No effect if not used
      with distributed training. See `tf.distribute.InputOptions` for more
      information.
  """

  def __init__(self, dataset_fn, input_options=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: dataset_fn, a callable taking an input context and returning a dataset;
    # input_options, optional distribution input options object (default nothing). Raises a type
    # error if dataset_fn is not callable, or if input_options is given (truthy) but is not an
    # input-options instance. Stores both as attributes; no other side effects.
    if not callable(dataset_fn):
      raise TypeError('`dataset_fn` for `DatasetCreator` must be a `callable`.')
    if input_options and (not isinstance(input_options,
                                         distribute_lib.InputOptions)):
      raise TypeError('`input_options` for `DatasetCreator` must be a '
                      '`tf.distribute.InputOptions`.')

    self.dataset_fn = dataset_fn
    self.input_options = input_options

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Invokes the stored dataset_fn, forwarding all positional and keyword arguments unchanged
    # (normally an input context). Returns the resulting dataset. Raises a type error if the
    # callable's result is not a dataset. No other side effects.
    # When a `DatasetCreator` is invoked, it forwards args/kwargs straight to
    # the callable.
    dataset = self.dataset_fn(*args, **kwargs)
    if not isinstance(dataset, data_types.DatasetV2):
      raise TypeError('The `callable` provided to `DatasetCreator` must return '
                      'a Dataset.')
    return dataset
