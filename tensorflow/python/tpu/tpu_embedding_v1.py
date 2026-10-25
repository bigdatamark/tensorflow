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
# Defines TPUEmbeddingV0, the mid level TPU embedding API that does not use embedding hardware.
# Exposes one exported class. Import-time side effect: export registration only.
"""Mid level API for TPU Embeddings without Embedding Accelerator."""

from typing import Any, Dict, Iterable, Optional, Text, Union

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("tpu.experimental.embedding.TPUEmbeddingV0")
class TPUEmbeddingV0(tpu_embedding_base.TPUEmbeddingBase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring example refers to an embedding_table attribute and mixes names;
  # the real accessor is the embedding_tables property.
  # Mid level embedding API for TPUs without embedding hardware: tables are replicated on every
  # device and lookups run as ordinary dense ops on the TPU core. Created under a TPU strategy
  # from a feature config and an optimizer; calls build lazily and exposes the table variables and
  # an embedding_lookup method handling dense, sparse and ragged inputs. Registered for export
  # as the public TPUEmbeddingV0 symbol. Holds the strategy and a built flag.
  """The TPUEmbedding mid level API running on TPU without Embedding accelerator.

  NOTE: This mid level API is not intended for large embedding table lookup.
  Embedding tables will be replicated across devices rather than sharding
  across them. To do large embedding table lookup, please use the
  `tpu.experimental.embedding.TPUEmbedding` class. This class is an alternative
  way to do embedding lookups when the TPU doesn't support any version of
  embedding feature. See
  `tpu.experimental.tpu_hardware_feature.embedding_feature` for a detailed
  explanation.

  This class has to be created under the `TPUStrategy`, Otherwise a RuntimeError
  will be raised.
  ```python
  strategy = tf.distribute.TPUStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbeddingV0(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```
  When creating a distributed dataset that is to be passed to the lookup
  operation a special input option must be specified:

  ```python
  distributed_dataset = (
      strategy.distribute_datasets_from_function(
          dataset_fn=...,
          options=tf.distribute.InputOptions(
              experimental_fetch_to_device=False))
  dataset_iterator = iter(distributed_dataset)
  ```

  Below is an example of a training and evaluation step:

  ```python
  optimizer = tf.keras.optimizers.SGD(0.1)

  @tf.function
  def training_step(dataset_iterator, num_steps):
    def tpu_step(embedding_features):
      with tf.GradientTape() as tape:
        tape.watch(embedding.embedding_table.values())
        activation = embedding(embedding_features)
        model_output = model(activations)
        loss = ...  # some function of labels and model_output

      embedding_gradients = tape.gradient(loss,
                                          embedding.embedding_table.values())
      optimizer.apply_gradients(list(zip(gradients,
                                mid_level_api.embedding_tables.values())))
      # Insert your model gradient and optimizer application here

    for _ in tf.range(num_steps):
      strategy.run(tpu_step, args=(next(dataset_iterator), ))

  @tf.function
  def evalution_step(dataset_iterator, num_steps):
    def tpu_step(embedding_features):
      activations = embedding(embedding_features)
      model_output = model(activations)
      # Insert your evaluation code here.

    for _ in tf.range(num_steps):
      strategy.run(tpu_step, args=(next(dataset_iterator), ))
  ```

  NOTE: The optimizer used here is a Keras optimizer. In order to make the slot
  variable creation stay consistent between Keras optimizers and
  embedding optimizers, the `slot_variable_creation_fn` argument of the
  embedding optimizers has to be passed with the Keras `add_slot` function. Also
  note that the slot names might be slightly different between them.

  ```python
  optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)

  def slot_variable_creation_fn(table, slot_names, slot_initializers):
      slots = {}
      for slot, initializer in zip(slot_names, slot_initializers):
        slots[slot] = optimizer.add_slot(table, slot, initializer)
      return slots

  embedding_optimizer = tf.experimental.embedding.Adagrad(
      learning_rate=0.1,
      slot_variable_creation_fn=slot_variable_creation_fn)

  # Use the embedding optimizer to create mid level api and keras optimizer to
  # apply gradients.
  ```
  """

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer]):  # pylint:disable=protected-access
    # [FUNCTION DOC]
    # Constructor. Inputs: feature_config is a feature config or nested structure of them;
    # optimizer is an embedding optimizer or nothing. Calls the base constructor, records the
    # current distribution strategy, and marks the object as not yet built.
    # Raises a runtime error if the current strategy is not a TPU strategy.
    super(TPUEmbeddingV0, self).__init__(feature_config, optimizer)
    self._strategy = distribute_lib.get_strategy()
    if not isinstance(self._strategy,
                      (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2)):
      raise RuntimeError(
          "TPUEmbeddingV0 should be created under TPUStrategy but found {}."
          .format(self._strategy))
    self._built = False

  @property
  def embedding_tables(
      self) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
    # [FUNCTION DOC]
    # Read-only property. Builds the variables if needed and returns a mapping from each table
    # config to its parameter variable (slot variables are excluded). Side effect: may build.
    """Returns a dict of embedding tables, keyed by `TableConfig`."""
    self._maybe_build()
    # Only return the tables and not the slot variables.
    return {
        table: self._variables[table.name]["parameters"]
        for table in self._table_config
    }

  def _create_variables_and_slots(
      self) -> Dict[Text, Dict[Text, tf_variables.Variable]]:
    # [FUNCTION DOC]
    # HELPER: variable creation hook called by the base class build step.
    # Creates, for every table config, a trainable parameter variable plus slot variables through
    # the base class helper, keyed by table name. Returns a mapping from table name to a mapping
    # of 'parameters' and slot names to variables. Side effect: creates variables.
    """Create variables for TPU embeddings.

    Note that this will always ensure that the variable is created under the
    TPUStrategy.

    Returns:
      A dict of dicts. The outer dict is keyed by the table names and the inner
      dicts are keyed by 'parameters' and the slot variable names.
    """
    variables = {}
    for table in self._table_config:
      # created TPUDistributedVariable.
      variables[table.name] = self._create_variables(table, trainable=True)
    return variables

  def _maybe_build(self):
    # [FUNCTION DOC]
    # HELPER: lazy-build guard for the embedding tables property.
    # Builds the variables once if not already built, inside an init scope so the build runs
    # eagerly even while tracing a function. Takes nothing, returns nothing.
    # The built flag is expected to be set by the base class build method.
    if not self._built:
      # This can be called while tracing a function, so we wrap the
      # initialization code with init_scope so it runs eagerly, this means that
      # it will not be included in the function graph generated by tracing so
      # that we can be sure that we only initialize the TPU for embeddings
      # exactly once.
      with ops.init_scope():
        self.build()

  def _apply_combiner_to_embeddings(
      self,
      embeddings: tensor.Tensor,
      weight: tensor.Tensor,
      combiner: Optional[Text] = None) -> tensor.Tensor:
    # [FUNCTION DOC]
    # HELPER: reduces looked-up embeddings over the second to last axis for lookups.
    # Inputs: embeddings (already multiplied by weights), weight tensor of matching layout, and
    # combiner name (sum, mean or sqrtn; nothing means mean). Sum adds the embeddings; mean divides
    # by the summed weights; sqrtn divides by the square root of the summed squared weights; the
    # divisions give zero where the denominator is zero. Returns the reduced tensor. Raises a
    # value error for any other combiner name. No side effects.
    """Apply the combiner to the embedding look up result on second to last axis.

    Args:
      embeddings: A Tensor of the embedding lookup result.
      weight: A Tensor of weight which has the same shape of the embeddings.
      combiner: One of "mean", "sum", "sqrtn". Defaults to "mean".

    Raises:
      ValueError: If the combiner is not one of 'mean', 'sqrtn' or 'sum'.
    Returns:
      A Tensor.
    """
    if combiner is None:
      combiner = "mean"
    if combiner == "sum":
      embeddings = math_ops.reduce_sum(embeddings, axis=-2)
    elif combiner == "mean":
      embeddings = math_ops.reduce_sum(embeddings, axis=-2)
      weight_sum = math_ops.reduce_sum(weight, axis=-2)
      embeddings = math_ops.div_no_nan(embeddings, weight_sum)
    elif combiner == "sqrtn":
      embeddings = math_ops.reduce_sum(embeddings, axis=-2)
      weight_squared = math_ops.pow(weight, 2)
      weight_sum = math_ops.reduce_sum(weight_squared, axis=-2)
      weight_sum_sqrt = math_ops.sqrt(weight_sum)
      embeddings = math_ops.div_no_nan(embeddings, weight_sum_sqrt)
    else:
      raise ValueError(
          f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")
    return embeddings

  def _pad_or_truncate_with_sequence_length(
      self, embeddings: tensor.Tensor, sequence_length: int
  ) -> tensor.Tensor:
    # [FUNCTION DOC]
    # HELPER: sequence-feature post-processing for the sparse lookup.
    # Makes the sequence axis (second axis) of a rank 3 embedding tensor exactly the requested
    # length: slices it if longer, zero pads at the end if shorter or equal. Inputs: embeddings
    # tensor and the target sequence length (integer). Returns the adjusted tensor. No side effects.
    """Pad or truncate the embedding lookup result based on the sequence length.

    Args:
      embeddings: A rank 3 Tensor of the embedding lookup result.
      sequence_length: number of the max sequence length set in the feature
        config.

    Returns:
      A Tensor with second last axis padded or truncated.
    """
    original_sequence_length = embeddings.shape[1]
    if original_sequence_length > sequence_length:
      embeddings = array_ops.slice(
          embeddings, begin=[0, 0, 0], size=[-1, sequence_length, -1])
    else:
      embeddings = array_ops.pad(
          embeddings,
          paddings=[[0, 0], [0, sequence_length - original_sequence_length],
                    [0, 0]])
    return embeddings

  def embedding_lookup(self,
                       features: Any,
                       weights: Optional[Any] = None) -> Any:
    # [FUNCTION DOC]
    # Public lookup entry point. Inputs: features is a nested structure (matching the feature
    # config) of dense, sparse or ragged tensors; weights is nothing or a structure matching
    # features whose entries may be nothing; weights must be of the same kind as their inputs.
    # Builds the tables if not yet built, checks the structures, then per feature: dense inputs
    # use a plain lookup; sparse and ragged inputs go through the dedicated helpers.
    # Returns a structure like the feature config holding the embedding results.
    # Raises a value error when weights are given for dense input, when weight type differs from
    # input type, when weights are given for a sequence feature, when a dense input is given for a
    # sequence feature, or when an input is not a tensor, sparse or ragged tensor; structure
    # mismatches raise from the nest utilities. Side effect: may create variables via build.
    """Apply embedding lookup on TPUs using Tensorcore.

    Note that all the sparse and ragged tensors will be converted to dense
    tensors on CPU and then passed to the TPU to do embedding look up. Large
    embedding lookup is not supported by this API, use the TPUEmbedding mid
    level api instead.

    Args:
      features: a nested structure of Tensors, SparseTensors or RaggedTensors.
      weights: a nested structure of Tensors, SparseTensors or RaggedTensors or
        None for no weights. If not None, structure must match that of inputs,
        but entries are allowed to be None.

    Returns:
      A nested structure of Tensors with the same structure as inputs.
    """
    if not self._built:
      self.build()
    nest.assert_same_structure(features, self._feature_config)

    flat_inputs = nest.flatten(features)
    flat_weights = [None] * len(flat_inputs)
    if weights is not None:
      nest.assert_same_structure(features, weights)
      flat_weights = nest.flatten(weights)
    flat_features = nest.flatten_with_joined_string_paths(self._feature_config)

    outputs = []
    for inp, weight, (path, feature) in zip(flat_inputs, flat_weights,
                                            flat_features):
      table = self.embedding_tables[feature.table]

      if weight is not None:
        if isinstance(inp, tensor.Tensor):
          raise ValueError(
              "Weight specified for {}, but input is dense.".format(path))
        elif type(weight) is not type(inp):
          raise ValueError(
              "Weight for {} is of type {} but it does not match type of the "
              "input which is {}.".format(path, type(weight), type(inp)))
        elif feature.max_sequence_length > 0:
          raise ValueError("Weight specified for {}, but this is a sequence "
                           "feature.".format(path))

      if isinstance(inp, tensor.Tensor):
        if feature.max_sequence_length > 0:
          raise ValueError(
              "Feature {} is a sequence feature but a dense tensor "
              "was passed.".format(path))
        outputs.append(embedding_ops.embedding_lookup_v2(table, inp))

      elif isinstance(inp, sparse_tensor.SparseTensor):
        outputs.append(
            self._embedding_lookup_for_sparse_tensor(inp, weight, table,
                                                     feature))
      elif isinstance(inp, ragged_tensor.RaggedTensor):
        outputs.append(
            self._embedding_lookup_for_ragged_tensor(inp, weight, table,
                                                     feature))
      else:
        raise ValueError("Input {} is type {}. Tensor, SparseTensor or "
                         "RaggedTensor expected.".format(path, type(inp)))
    return nest.pack_sequence_as(self._feature_config, outputs)

  def _embedding_lookup_for_sparse_tensor(
      self, inp: sparse_tensor.SparseTensor,
      weight: Optional[sparse_tensor.SparseTensor],
      table: tf_variables.Variable,
      feature: tpu_embedding_v2_utils.FeatureConfig) -> tensor.Tensor:
    # [FUNCTION DOC]
    # HELPER: per-feature lookup used by embedding_lookup for sparse inputs.
    # Looks up embeddings for a sparse id tensor. Inputs: inp is the sparse ids; weight is a
    # sparse weight tensor or nothing (unit weights); table is the embedding table variable;
    # feature is the feature config (combiner, output shape, max sequence length).
    # Steps: densifies ids and weights on the host via outside compilation, gathers embeddings,
    # multiplies by the weights, then pads or truncates along the sequence axis for sequence
    # features without an output shape, otherwise applies the table combiner over the second to
    # last axis. Returns the embedding tensor. Raises nothing itself (the combiner helper may
    # raise a value error for an unknown combiner). Side effect: outside compilation on the TPU.
    """Embedding lookup for sparse tensor based on its feature config.

    Args:
      inp: a single SparseTensor input.
      weight: None or SparseTensor which has the same shape of the input.
      table: a table variable.
      feature: a feature config.

    Returns:
      Embedding lookup result.
    """

    # This computation needs to placed outside of tpu as the size of the
    # indices and values can change for different batch which can cause
    # the program to re-compile.
    def sparse_to_dense_computation(inp, weight):
      # [FUNCTION DOC]
      # HELPER: runs on the host (outside TPU compilation) for the sparse lookup helper.
      # Converts a sparse id tensor and an optional sparse weight tensor into dense tensors.
      # Inputs: inp is the sparse id tensor; weight is a sparse tensor of the same layout or
      # nothing, in which case weights of one (float32) are created at the input's indices.
      # Returns a pair of dense tensors (ids, weights), padded with zeros where entries are
      # missing. Raises nothing itself; has no side effects.
      if weight is None:
        weight = sparse_tensor.SparseTensor(
            inp.indices,
            array_ops.ones_like(inp.values, dtype=dtypes.float32),
            dense_shape=inp.dense_shape)
      # Pad the sparse tensor to be dense tensor.
      inp = sparse_ops.sparse_tensor_to_dense(inp)
      weight = sparse_ops.sparse_tensor_to_dense(weight)
      return inp, weight

    inp, weight = tpu_replication.outside_compilation(
        sparse_to_dense_computation, inp=inp, weight=weight)

    embeddings = embedding_ops.embedding_lookup_v2(table, inp)
    weight = array_ops.expand_dims(weight, -1)
    embeddings *= weight
    if not feature.output_shape and feature.max_sequence_length > 0:
      embeddings = self._pad_or_truncate_with_sequence_length(
          embeddings, feature.max_sequence_length)
    else:
      embeddings = self._apply_combiner_to_embeddings(embeddings, weight,
                                                      feature.table.combiner)
    return embeddings

  def _embedding_lookup_for_ragged_tensor(
      self, inp: ragged_tensor.RaggedTensor,
      weight: Optional[ragged_tensor.RaggedTensor],
      table: tf_variables.Variable,
      feature: tpu_embedding_v2_utils.FeatureConfig) -> tensor.Tensor:
    # [FUNCTION DOC]
    # HELPER: per-feature lookup used by embedding_lookup for ragged inputs.
    # KNOWN ISSUE: the docstring says the output shape is checked against the first dim of the
    # input, but the real check requires the output shape element count to equal or be a multiple
    # of the batch size.
    # Looks up embeddings for a rank 2 ragged id tensor. Inputs: inp is the ragged ids; weight is
    # a ragged weight tensor or nothing; table is the embedding table variable; feature is the
    # feature config (table combiner and dim, output shape, max sequence length).
    # Steps: validates rank, converts to dense on the host via outside compilation, gathers
    # embeddings, multiplies by weights, then either combines over the second to last axis
    # (default, or when output shape element count equals the batch size) or leaves the sequence
    # axis in place (sequence features). When an output shape is set the result is reshaped to
    # that shape plus the embedding dimension.
    # Returns the dense embedding tensor. Raises a value error if the input is not rank 2 or the
    # output shape is not compatible with the batch size (raised from the nested converter).
    # Side effects: eager evaluation inside an init scope; outside compilation on the TPU.
    """Embedding lookup for ragged tensor based on its feature config.

    Args:
      inp: a single rank 2 RaggedTensor input.
      weight: None or RaggedTensor which has the same shape of the input.
      table: a table variable.
      feature: a feature config.

    Returns:
      Embedding lookup result.

    Raises:
      ValueError: if input ragged tensor is not rank 2 or output shape set in
      the feature config doesn't match with the first dim size of the input.
    """
    if inp.shape.rank != 2:
      raise ValueError(
          "Only rank 2 ragged tensor is supported, but got rank {}".format(
              inp.shape.rank))
    batch_size = inp.shape[0]

    # This computation needs to placed outside of tpu as the size of the row
    # splits and values can change for different batch which can cause
    # the program to re-compile.
    def ragged_to_dense_outside_compilation(inp, weight, batch_size, feature):
      # [FUNCTION DOC]
      # HELPER: runs on the host (outside TPU compilation) for the ragged lookup helper.
      # Converts a rank 2 ragged input and its optional ragged weight into dense tensors.
      # Inputs: inp is the ragged id tensor; weight is a ragged weight tensor or nothing (then
      # weights of one are created with the same row splits); batch_size is the static first
      # dimension of the input; feature is the feature config (output shape, max sequence length).
      # Behavior: for a sequence feature without output shape, the input is padded or truncated to
      # batch size by max sequence length and weights become all ones. If an output shape is set,
      # its element count is computed eagerly; equal to batch size means normal dense conversion;
      # a larger exact multiple means pad/truncate to the quotient as sequence length with unit
      # weights. Otherwise the plain dense conversion of input and weight is returned.
      # Returns a pair of dense tensors (ids, weights). Raises a value error if the output shape
      # element count is not a multiple of the batch size (and not equal to it).
      # Side effect: evaluates the output shape product eagerly inside an init scope.
      if weight is None:
        weight = ragged_tensor.RaggedTensor.from_row_splits(
            array_ops.ones_like(inp.values, dtype=dtypes.float32),
            inp.row_splits)
      if not feature.output_shape and feature.max_sequence_length > 0:
        inp = inp.to_tensor(shape=(batch_size, feature.max_sequence_length))
        # Ignore weight if it is a sequence feature.
        weight = array_ops.ones_like(inp, dtype=dtypes.float32)
      elif feature.output_shape:
        # Eagerly run the following op as the result as to be a number in
        # order to use it as part of the output shape.
        with ops.init_scope():
          output_batch_size = math_ops.reduce_prod(feature.output_shape).numpy()
        # If the output batch size matches the data batch size, treat it as
        # normal ragged input.
        if output_batch_size == batch_size:
          inp, weight = inp.to_tensor(), weight.to_tensor()
        # If the data batch size is a factor of the output batch size, the
        # divide result will be the sequence length. Ignore the weights and
        # combiner.
        elif (
            output_batch_size > batch_size
            and output_batch_size % batch_size == 0
        ):
          # Pad or truncate in the sequence dimension
          seq_length = output_batch_size // batch_size
          inp = inp.to_tensor(shape=(batch_size, seq_length))
          # Ignore weight if it is a sequence feature.
          weight = array_ops.ones_like(inp, dtype=dtypes.float32)
        else:
          raise ValueError(
              "Output shape set in the FeatureConfig should be the factor of "
              "the input data batch size. But instead got output shape {}, "
              "input data batch size {}".format(feature.output_shape,
                                                batch_size))
      else:
        inp, weight = inp.to_tensor(), weight.to_tensor()
      return inp, weight

    inp, weight = tpu_replication.outside_compilation(
        ragged_to_dense_outside_compilation,
        inp=inp,
        weight=weight,
        batch_size=batch_size,
        feature=feature)

    embeddings = embedding_ops.embedding_lookup_v2(table, inp)
    weight = array_ops.expand_dims(weight, -1)
    embeddings *= weight

    if feature.output_shape:
      with ops.init_scope():
        output_batch_size = math_ops.reduce_prod(feature.output_shape).numpy()
      if output_batch_size == batch_size:
        embeddings = self._apply_combiner_to_embeddings(embeddings, weight,
                                                        feature.table.combiner)
      embeddings = array_ops.reshape(
          embeddings, shape=feature.output_shape + [feature.table.dim])
    else:
      if feature.max_sequence_length == 0:
        embeddings = self._apply_combiner_to_embeddings(embeddings, weight,
                                                        feature.table.combiner)
    return embeddings
