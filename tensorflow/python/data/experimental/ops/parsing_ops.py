# [MODULE DOC]
# Experimental tf.data transformation for parsing serialized Example protocol buffers inside a
# dataset pipeline. Exposes parse_example_dataset (deprecated in favor of mapping the standard
# parse function) and a private dataset class implementing it via the generated parse-example
# dataset operation (version 2). Sparse and ragged features are post-processed with a map step
# that builds composite tensors.
# Import-time side effect: the public function is registered for export under the experimental
# data API.
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
"""Experimental `dataset` API for parsing example."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


class _ParseExampleDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: private dataset implementation behind parse_example_dataset.
  # Unary dataset that takes a dataset of vectors of serialized example strings and produces a
  # dictionary per element mapping feature keys to dense, sparse or ragged tensors. Holds the
  # feature configuration, parallelism, determinism mode string and the computed element
  # structure dictionary.
  """A `Dataset` that parses `example` dataset into a `dict` dataset."""

  def __init__(self, input_dataset, features, num_parallel_calls,
               deterministic):
    # [FUNCTION DOC]
    # Builds the dataset. Inputs: input_dataset, whose elements must be vectors of strings
    # (raises a type error otherwise); features, mapping from keys to feature specifications
    # (fixed-length, fixed-length sequence, variable-length, sparse or ragged); num_parallel_calls,
    # number of parallel parsing calls; deterministic, nothing/true/false mapped to the strings
    # default, true or false. Prepends an unknown leading dimension to the features, extracts
    # parsing parameters, then fills the element structure: sparse tensor specs for sparse keys,
    # tensor specs for dense keys (input shape plus dense shape), ragged specs for ragged keys.
    # Creates the version 2 parse-example dataset operation and calls the parent constructor.
    # Uses private helpers of the parsing module.
    self._input_dataset = input_dataset
    if not structure.are_compatible(
        input_dataset.element_spec,
        tensor_spec.TensorSpec([None], dtypes.string)):
      raise TypeError("Input dataset should be a dataset of vectors of "
                      f"strings. Instead it is `{input_dataset.element_spec}`.")
    self._num_parallel_calls = num_parallel_calls
    if deterministic is None:
      self._deterministic = "default"
    elif deterministic:
      self._deterministic = "true"
    else:
      self._deterministic = "false"
    # pylint: disable=protected-access
    self._features = parsing_ops._prepend_none_dimension(features)
    params = parsing_ops._ParseOpParams.from_features(self._features, [
        parsing_ops.VarLenFeature, parsing_ops.SparseFeature,
        parsing_ops.FixedLenFeature, parsing_ops.FixedLenSequenceFeature,
        parsing_ops.RaggedFeature
    ])
    # pylint: enable=protected-access
    self._sparse_keys = params.sparse_keys
    self._sparse_types = params.sparse_types
    self._ragged_keys = params.ragged_keys
    self._ragged_value_types = params.ragged_value_types
    self._ragged_split_types = params.ragged_split_types
    self._dense_keys = params.dense_keys
    self._dense_defaults = params.dense_defaults_vec
    self._dense_shapes = params.dense_shapes_as_proto
    self._dense_types = params.dense_types
    input_dataset_shape = dataset_ops.get_legacy_output_shapes(
        self._input_dataset)

    self._element_spec = {}

    for (key, value_type) in zip(params.sparse_keys, params.sparse_types):
      self._element_spec[key] = sparse_tensor.SparseTensorSpec(
          input_dataset_shape.concatenate([None]), value_type)

    for (key, value_type, dense_shape) in zip(params.dense_keys,
                                              params.dense_types,
                                              params.dense_shapes):
      self._element_spec[key] = tensor_spec.TensorSpec(
          input_dataset_shape.concatenate(dense_shape), value_type)

    for (key, value_type, splits_type) in zip(params.ragged_keys,
                                              params.ragged_value_types,
                                              params.ragged_split_types):
      self._element_spec[key] = ragged_tensor.RaggedTensorSpec(
          input_dataset_shape.concatenate([None]), value_type, 1, splits_type)

    variant_tensor = (
        gen_experimental_dataset_ops.parse_example_dataset_v2(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._num_parallel_calls,
            self._dense_defaults,
            self._sparse_keys,
            self._dense_keys,
            self._sparse_types,
            self._dense_shapes,
            deterministic=self._deterministic,
            ragged_keys=self._ragged_keys,
            ragged_value_types=self._ragged_value_types,
            ragged_split_types=self._ragged_split_types,
            **self._flat_structure))
    super(_ParseExampleDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the dictionary of per-key tensor structure specs computed during
    # construction.
    return self._element_spec


@tf_export("data.experimental.parse_example_dataset")
@deprecation.deprecated(
    None, "Use `tf.data.Dataset.map(tf.io.parse_example(...))` instead.")
def parse_example_dataset(features, num_parallel_calls=1, deterministic=None):
  # [FUNCTION DOC]
  # Deprecated. Creates a transformation parsing serialized example protos into dictionaries of
  # tensors. Inputs: features, required mapping from keys to feature specifications (raises a
  # value error immediately if nothing); num_parallel_calls, number of parallel parse calls
  # (default 1); deterministic, optional boolean allowing out-of-order output when false
  # (nothing follows the dataset option). Returns a function that, applied to a dataset of
  # vectors of serialized strings, yields the parsed dataset.
  """A transformation that parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized `Example` protos given in `serialized`. We refer
  to `serialized` as a batch with `batch_size` many entries of individual
  `Example` protos.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`,
  `SparseTensor`, and `RaggedTensor` objects. `features` is a dict from keys to
  `VarLenFeature`, `RaggedFeature`, `SparseFeature`, and `FixedLenFeature`
  objects. Each `VarLenFeature` and `SparseFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and each
  `FixedLenFeature` is mapped to a `Tensor`. See `tf.io.parse_example` for more
  details about feature dictionaries.

  Args:
   features: A `dict` mapping feature keys to `FixedLenFeature`,
     `VarLenFeature`, `RaggedFeature`, and `SparseFeature` values.
   num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
      representing the number of parsing processes to call in parallel.
   deterministic: (Optional.) A boolean controlling whether determinism
      should be traded for performance by allowing elements to be produced out
      of order if some parsing calls complete faster than others. If
      `deterministic` is `None`, the
      `tf.data.Options.deterministic` dataset option (`True` by default) is used
      to decide whether to produce elements deterministically.

  Returns:
    A dataset transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if features argument is None.
  """
  if features is None:
    raise ValueError("Argument `features` is required, but not specified.")

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: closure returned by parse_example_dataset.
    # Wraps the dataset in the private parse dataset. If any feature is sparse or ragged, adds a
    # map step (with the same parallelism) that builds composite tensors from the raw parsed
    # components. Returns the resulting dataset.
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    out_dataset = _ParseExampleDataset(dataset, features, num_parallel_calls,
                                       deterministic)
    if any(
        isinstance(feature, parsing_ops.SparseFeature) or
        isinstance(feature, parsing_ops.RaggedFeature)
        for feature in features.values()):
      # pylint: disable=protected-access
      # pylint: disable=g-long-lambda
      out_dataset = out_dataset.map(
          lambda x: parsing_ops._construct_tensors_for_composite_features(
              features, x),
          num_parallel_calls=num_parallel_calls)
    return out_dataset

  return _apply_fn
