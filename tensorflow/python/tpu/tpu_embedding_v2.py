# [MODULE DOC]
# Mid level (v2) API for TPU embeddings. Exposes the public TPUEmbedding trackable class
# (exported as the experimental TPU embedding API), plus helpers used to create sharded
# embedding variables across TPU hosts and to load/retrieve table values to/from the TPU.
# The class builds a TPU embedding configuration from user feature and table configs,
# creates (sharded) table and optimizer slot variables, and provides enqueue, dequeue and
# apply_gradients operations for use inside custom training loops under a TPU strategy.
# Under a non-TPU strategy it only exposes the table variables, for CPU serving.
# Import-time side effects: registers a checkpoint saver ("TPUEmbeddingCallback") so that
# saving retrieves variables from the TPU and restoring loads them back onto the TPU.
# Also relies on the sibling utils module for feature, table and optimizer configuration.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Mid level API for TPU Embeddings."""

import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

from absl import logging

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


_HOOK_KEY = "TPUEmbedding_saveable"
_NAME_KEY = "_tpu_embedding_layer"


class TPUEmbeddingVariable(sharded_variable.ShardedVariableMixin):
  # [CLASS DOC]
  # Sharded variable type used for TPU embedding tables and slot variables. It holds a list
  # of per-host variable shards (created by the sharded variable creator) and behaves as a
  # single logical variable through the sharded-variable mixin. Created by
  # make_sharded_variable_creator; adds only the graph-mode query below.
  """A ShardedVariable class for TPU."""

  @property
  def _in_graph_mode(self):
    # [FUNCTION DOC]
    # HELPER: lets the sharded-variable machinery ask whether the variable is graph based.
    # Property returning whether the first shard variable is in graph mode (a boolean).
    # Takes no inputs and has no side effects; assumes at least one shard exists, otherwise
    # indexing the empty shard list fails.
    return self.variables[0]._in_graph_mode  # pylint: disable=protected-access


def _add_key_attr(op, name):
  # [FUNCTION DOC]
  # HELPER: tags enqueue, dequeue and apply-gradients ops with the user supplied name.
  # Sets a private string attribute on the given operation holding the given name (text,
  # converted to bytes). Mutates the operation in place; returns nothing.
  op._set_attr(_NAME_KEY, attr_value_pb2.AttrValue(s=compat.as_bytes(name)))  # pylint: disable=protected-access


@tf_export("tpu.experimental.embedding.TPUEmbedding")
class TPUEmbedding(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # Public mid level TPU embedding API (trackable, so it can be checkpointed). Created from
  # a nested structure of feature configs and an optimizer, optionally under a TPU strategy.
  # State: the strategy, whether TPU is used, the output shapes, the ordered list of unique
  # tables, dynamic (callable) learning rates, host devices, the config proto, the created
  # variables (mapping from table name to parameters and slot variables) and a built flag.
  # Under a TPU strategy it supports build, enqueue, dequeue and apply_gradients; otherwise
  # only embedding_tables (for CPU lookups). Variables are lazily created in build, which
  # also initializes the TPU embedding engine. Saving and restoring go through a registered
  # checkpoint saver that retrieves from or loads to the TPU.
  """The TPUEmbedding mid level API.

  NOTE: When instantiated under a TPUStrategy, this class can only be created
  once per call to `tf.tpu.experimental.initialize_tpu_system`. If you wish to
  re-initialize the embedding engine you must re-initialize the tpu as well.
  Doing this will clear any variables from TPU, so ensure you have checkpointed
  before you do this. If a further instances of the class are needed,
  set the `initialize_tpu_embedding` argument to `False`.

  This class can be used to support training large embeddings on TPU. When
  creating an instance of this class, you must specify the complete set of
  tables and features you expect to lookup in those tables. See the
  documentation of `tf.tpu.experimental.embedding.TableConfig` and
  `tf.tpu.experimental.embedding.FeatureConfig` for more details on the complete
  set of options. We will cover the basic usage here.

  NOTE: multiple `FeatureConfig` objects can use the same `TableConfig` object,
  allowing different features to share the same table:

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  ```

  There are two modes under which the `TPUEmbedding` class can used. This
  depends on if the class was created under a `TPUStrategy` scope or not.

  Under `TPUStrategy`, we allow access to the method `enqueue`, `dequeue` and
  `apply_gradients`. We will show examples below of how to use these to train
  and evaluate your model. Under CPU, we only access to the `embedding_tables`
  property which allow access to the embedding tables so that you can use them
  to run model evaluation/prediction on CPU.

  First lets look at the `TPUStrategy` mode. Initial setup looks like:

  ```python
  strategy = tf.distribute.TPUStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  When creating a distributed dataset that is to be passed to the enqueue
  operation a special input option must be specified:

  ```python
  distributed_dataset = (
      strategy.distribute_datasets_from_function(
          dataset_fn=...,
          options=tf.distribute.InputOptions(
              experimental_fetch_to_device=False))
  dataset_iterator = iter(distributed_dataset)
  ```

  Different feature inputs can have different shapes. For dense and sparse
  tensor, rank 2 and above is supported. For ragged tensor, although only rank 2
  is supported, you can specify the output shape to be rank 2 and above. The
  output shape specified in the FeatureConfig has the first priority. The input
  shape passed in build method has second priority and the input shapes
  auto detected from input feature has the lowest priority. The latter two will
  be converted to output shapes by omitting the last dimension. If the lower
  priority one has output shapes which don't match the former one. A ValueError
  will be raised. Only when the former one has undefined output shapes, the
  latter one can override.

  NOTE: All batches passed to the layer can have different input shapes. But
  these input shapes need to match with the output shapes set by either
  `FeatureConfig` or build method except for ragged tensor. Only 2D
  ragged tensor with output shape set to higher dimensions is allowed as
  long as the total number of elements matches. All subsequent calls must have
  the same input shapes. In the event that the input shapes cannot be
  automatically determined by the enqueue method, you must call
  the build method with the input shapes or provide output shapes in the
  `FeatureConfig` to initialize the layer.

  To use this API on TPU you should use a custom training loop. Below is an
  example of a training and evaluation step:

  ```python
  @tf.function
  def training_step(dataset_iterator, num_steps):
    def tpu_step(tpu_features):
      with tf.GradientTape() as tape:
        activations = embedding.dequeue()
        tape.watch(activations)
        model_output = model(activations)
        loss = ...  # some function of labels and model_output

      embedding_gradients = tape.gradient(loss, activations)
      embedding.apply_gradients(embedding_gradients)
      # Insert your model gradient and optimizer application here

    for _ in tf.range(num_steps):
      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(tpu_features, ))

  @tf.function
  def evaluation_step(dataset_iterator, num_steps):
    def tpu_step(tpu_features):
      activations = embedding.dequeue()
      model_output = model(activations)
      # Insert your evaluation code here.

    for _ in tf.range(num_steps):
      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=False)
      strategy.run(tpu_step, args=(tpu_features, ))
  ```

  NOTE: The calls to `enqueue` have `training` set to `True` when
  `embedding.apply_gradients` is used and set to `False` when
  `embedding.apply_gradients` is not present in the function. If you don't
  follow this pattern you may cause an error to be raised or the tpu may
  deadlock.

  In the above examples, we assume that the user has a dataset which returns
  a tuple where the first element of the tuple matches the structure of what
  was passed as the `feature_config` argument to the object initializer. Also we
  utilize `tf.range` to get a `tf.while_loop` in order to increase performance.

  When checkpointing your model, you should include your
  `tf.tpu.experimental.embedding.TPUEmbedding` object in the checkpoint. It is a
  trackable object and saving it will save the embedding tables and their
  optimizer slot variables:

  ```python
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.save(...)
  ```

  On CPU, only the `embedding_table` property is usable. This will allow you to
  restore a checkpoint to the object and have access to the table variables:

  ```python
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  tables = embedding.embedding_tables
  ```

  You can now use table in functions like `tf.nn.embedding_lookup` to perform
  your embedding lookup and pass to your model.

  """

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer],  # pylint:disable=protected-access
      pipeline_execution_with_tensor_core: bool = False):
    # [FUNCTION DOC]
    # Constructor. Inputs: feature_config, a nested structure of feature configs; optimizer,
    # an optimizer config applied to every table without its own optimizer (may be nothing
    # when not under a TPU strategy); pipeline_execution_with_tensor_core, a flag (default
    # false) to overlap embedding work with tensor core work. Records the current strategy
    # and whether it is a TPU strategy, the per-feature output shapes, and the number of
    # cores per replica (from the device assignment, if any). Collects the unique tables in
    # order of first use, assigns the default optimizer to tables lacking one (this mutates
    # the user's table configs) and generates names "table_N" for unnamed tables. Under TPU,
    # also collects distinct callable learning rates and the list of host devices.
    # Raises a value error for an unsupported optimizer (when one is needed) or for
    # duplicate table names. Does not create variables; marks the object as not built.
    """Creates the TPUEmbedding mid level API object.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(
          feature_config=tf.tpu.experimental.embedding.FeatureConfig(
              table=tf.tpu.experimental.embedding.TableConfig(
                  dim=...,
                  vocabulary_size=...)))
    ```

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. When not created under
        TPUStrategy may be set to None to avoid the creation of the optimizer
        slot variables, useful for optimizing memory consumption when exporting
        the model for serving where slot variables aren't needed.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old). Set to True for improved performance.

    Raises:
      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,
      Adam or Adagrad) or None when created under a TPUStrategy.
    """
    self._strategy = distribute_lib.get_strategy()
    self._using_tpu = isinstance(self._strategy, (tpu_strategy.TPUStrategy,
                                                  tpu_strategy.TPUStrategyV2))
    self._pipeline_execution_with_tensor_core = (
        pipeline_execution_with_tensor_core)

    self._feature_config = feature_config
    self._output_shapes = []
    for feature in nest.flatten(feature_config):
      self._output_shapes.append(feature.output_shape)

    device_assignment = getattr(
        self._strategy.extended, "_device_assignment", None
    )
    self._num_cores_per_replica = (
        device_assignment.num_cores_per_replica if device_assignment else None
    )

    # The TPU embedding ops are slightly inconsistent with how they refer to
    # tables:
    # * The enqueue op takes a parallel list of tensors for input, one of those
    #   is the table id for the feature which matches the integer index of the
    #   table in the proto created by _create_config_proto().
    # * The recv_tpu_embedding_activations op emits lookups per table in the
    #   order from the config proto.
    # * The send_tpu_embedding_gradients expects input tensors to be per table
    #   in the same order as the config proto.
    # * Per optimizer load and retrieve ops are specified per table and take the
    #   table name rather than the table id.
    # Thus we must fix a common order to tables and ensure they have unique
    # names.

    # Set table order here to the order of the first occurence of the table in a
    # feature provided by the user. The order of this struct must be fixed
    # to provide the user with deterministic behavior over multiple
    # instantiations.
    self._table_config = []
    for feature in nest.flatten(feature_config):
      if feature.table not in self._table_config:
        self._table_config.append(feature.table)

    # Ensure tables have unique names. Also error check the optimizer as we
    # specifically don't do that in the TableConfig class to allow high level
    # APIs that are built on this to use strings/other classes to represent
    # optimizers (before they are passed to this class).
    table_names = []
    for i, table in enumerate(self._table_config):
      if table.optimizer is None:
        # TODO(bfontain) Should we allow some sort of optimizer merging here?
        table.optimizer = optimizer
      if ((table.optimizer is not None or self._using_tpu) and
          not isinstance(table.optimizer, tpu_embedding_v2_utils._Optimizer)):  # pylint: disable=protected-access
        raise ValueError("{} is an unsupported optimizer class. Please pass an "
                         "instance of one of the optimizer classes under "
                         "tf.tpu.experimental.embedding.".format(
                             type(table.optimizer)))
      if table.name is None:
        table.name = "table_{}".format(i)
      if table.name in table_names:
        raise ValueError("Tables must have a unique name. "
                         f"Multiple tables with name {table.name} found.")
      table_names.append(table.name)

    if self._using_tpu:
      # Extract a list of callable learning rates also in fixed order. Each
      # table in the config proto will get an index into this list, and we will
      # pass this list in the same order after evaluation to the
      # send_tpu_embedding_gradients op.
      self._dynamic_learning_rates = []
      for table in self._table_config:
        if (callable(table.optimizer.learning_rate) and
            table.optimizer.learning_rate not in self._dynamic_learning_rates):
          self._dynamic_learning_rates.append(table.optimizer.learning_rate)

      # We need to list of host devices for the load/retrieve operations.
      self._hosts = tpu_embedding_v2_utils.get_list_of_hosts(self._strategy)

    self._built = False
    self._verify_output_shapes_on_enqueue = True

  def build(self, per_replica_input_shapes=None, per_replica_batch_size=None):  # pylint:disable=g-bare-generic
    # [FUNCTION DOC]
    # Creates the table and slot variables and, under a TPU strategy, initializes the TPU
    # embedding engine. Inputs: per_replica_input_shapes, optional nested structure of shapes
    # matching the feature config (or, deprecated, a plain integer batch size);
    # per_replica_batch_size, optional deprecated batch size. Does nothing if already built.
    # Under TPU: raises a runtime error if the embedding engine is already initialized,
    # resolves and validates output shapes, builds the config proto, logs it and runs a
    # traced function that initializes the engine. Then creates variables, marks the object
    # built and loads variable values onto the TPU. May raise a value error for inconsistent
    # or not fully defined output shapes. Returns nothing.
    """Create the underlying variables and initializes the TPU for embeddings.

    This method creates the underlying variables (including slot variables). If
    created under a TPUStrategy, this will also initialize the TPU for
    embeddings.

    This function will automatically get called by enqueue, which will try to
    determine your output shapes. If this fails, you must manually
    call this method before you call enqueue.

    Args:
      per_replica_input_shapes: A nested structure of The per replica input
        shapes that matches the structure of the feature config. The input
        shapes should be the same as the input shape of the feature (except for
        ragged tensor) Note that it is fixed and the same per replica input
        shapes must be used for both training and evaluation. If you want to
        calculate this from the global input shapes, you can use
        `num_replicas_in_sync` property of your strategy object. May be set to
        None if not created under a TPUStrategy.
      per_replica_batch_size: (Deprecated) The per replica batch size that you
        intend to use. Note that is fixed and the same batch size must be used
        for both training and evaluation. If you want to calculate this from the
        global batch size, you can use `num_replicas_in_sync` property of your
        strategy object. May be set to None if not created under a TPUStrategy.

    Raises:
      ValueError: If per_replica_input_shapes is inconsistent with the output
      shapes stored in the feature config or the output shapes get from the
      input shapes are not fully defined.
      RuntimeError: If tpu embedding is already initialized on TPU.
    """
    if self._built:
      return

    if self._using_tpu:
      # If the tpu embedding is already initialized on TPU, raise runtime error.
      # Below logic is not added in `initialize_system_for_tpu_embedding`
      # because doing exception control flow in graph mode is difficult.
      if tpu_ops.is_tpu_embedding_initialized():
        raise RuntimeError(
            "TPU is already initialized for embeddings. This may be caused by "
            "using multiple TPUEmbedding instances in a TPU scope which is "
            "unsupported")
      self._get_and_update_output_shapes_from_input(per_replica_input_shapes,
                                                    per_replica_batch_size)

      self._config_proto = self._create_config_proto()

      logging.info("Initializing TPU Embedding engine.")
      tpu_embedding_v2_utils.log_tpu_embedding_configuration(self._config_proto)

      @def_function.function
      def load_config():
        # [FUNCTION DOC]
        # Nested helper of build: traced function that initializes the TPU system for
        # embeddings using the already created config proto. Takes no inputs, returns nothing;
        # side effect is configuring the TPU embedding engine.
        tpu.initialize_system_for_tpu_embedding(self._config_proto)

      load_config()
      logging.info("Done initializing TPU Embedding engine.")

    # Create and load variables and slot variables into the TPU.
    # Note that this is a dict of dicts. Keys to the first dict are table names.
    # We would prefer to use TableConfigs, but then these variables won't be
    # properly tracked by the tracking API.
    self._variables = self._create_variables_and_slots()

    self._built = True

    # This is internally conditioned self._built and self._using_tpu
    self._load_variables()

  def _maybe_build(self,
                   output_shapes: Optional[Union[List[int], Iterable]] = None):  # pylint:disable=g-bare-generic
    # [FUNCTION DOC]
    # HELPER: lazily builds the object for enqueue and embedding_tables.
    # If not yet built, calls build with the given shapes inside an init scope so that the
    # initialization runs eagerly once even when invoked while tracing a function.
    # KNOWN ISSUE: the parameter is named output_shapes but it is forwarded to build as the
    # per-replica input shapes (enqueue passes input shapes), so the name is misleading.
    # Input: optional shapes. Returns nothing; does nothing when already built.
    if not self._built:
      # This can be called while tracing a function, so we wrap the
      # initialization code with init_scope so it runs eagerly, this means that
      # it will not be included the function graph generated by tracing so that
      # we can be sure that we only initialize the TPU for embeddings exactly
      # once.
      with ops.init_scope():
        self.build(output_shapes)

  def _get_and_update_output_shapes_from_input(
      self,
      per_replica_input_shapes: Optional[List[TensorShape]] = None,
      per_replica_batch_size: Optional[int] = None):
    # [FUNCTION DOC]
    # HELPER: part of build; resolves the per replica output shapes from user supplied input.
    # Inputs: optional per replica input shapes (nested structure, or a plain integer batch
    # size which is deprecated) and optional deprecated batch size. Logs deprecation
    # warnings, converts the given input or batch size to output shapes, checks them against
    # the existing output shapes (value error on mismatch), merges them into the stored
    # output shapes (existing ones win) and finally requires all output shapes to be fully
    # defined (value error otherwise). Mutates the stored output shapes; returns nothing.
    """Get and update the per replica output shapes from the input."""
    per_replica_output_shapes = None
    if per_replica_batch_size and per_replica_input_shapes is None:
      logging.warning(
          "per_replica_batch_size argument will be deprecated, please specify "
          "all the input shapes using per_replica_input_shapes argument.")
      per_replica_output_shapes = self._get_output_shapes_from_batch_size(
          per_replica_batch_size)

    # Update the input shapes if provided.
    if per_replica_input_shapes is not None:
      if isinstance(per_replica_input_shapes, int):
        logging.warning(
            "Passing batch size to per_replica_input_shapes argument will be"
            " deprecated, please specify all the input shapes using"
            " per_replica_input_shapes argument.")
        per_replica_output_shapes = self._get_output_shapes_from_batch_size(
            per_replica_input_shapes)
      else:
        nest.assert_same_structure(
            nest.flatten(per_replica_input_shapes),
            nest.flatten(self._feature_config))

        # Convert the nested structure to list.
        per_replica_input_shapes = nest.flatten(per_replica_input_shapes)

        per_replica_output_shapes = self._get_output_shapes_from_input_shapes(
            per_replica_input_shapes)

    if per_replica_output_shapes is not None:

      # Check the output shapes with existing output shapes setting.
      self._check_output_shapes(per_replica_output_shapes)

      # Update the output shapes with existing output shapes setting.
      # This is necessary Because the output shapes might be missing from
      # the feature config, the usr can set it:
      #  1. calling the build method
      #  2. output shapes auto detected when calling the dequeue method for
      #     for the first time. The dequeue method will call build method
      #     with the output shapes.
      # Either these two situations will lead to an update to the existing
      # output shapes.
      self._update_output_shapes(per_replica_output_shapes)

    # Check if the output shapes are fully defined. This is required in order
    # to set them in the feature descriptor field of the tpu embedding config
    # proto.
    self._check_output_shapes_fully_defined()

  def _get_output_shapes_from_input_shapes(
      self, input_shapes: List[TensorShape]) -> List[TensorShape]:
    # [FUNCTION DOC]
    # HELPER: converts flat per feature input shapes into output shapes.
    # Input: list of input shapes in feature order. For each, raises a value error when the
    # rank is unknown or below one. For rank 2 shapes whose last dimension is not one, with
    # no output shape in the feature config and a positive max sequence length, inserts the
    # sequence length as the second to last dimension. Rank one shapes are kept as is; others
    # drop their last dimension. Returns the list of output shapes; no side effects.
    """Get output shapes from the flattened input shapes list."""
    output_shapes = []
    for input_shape, feature in zip(input_shapes,
                                    nest.flatten(self._feature_config)):
      if input_shape.rank is None or input_shape.rank < 1:
        raise ValueError(
            "Received input tensor of shape {}. Rank must be 1 and above"
            .format(input_shape))
      # Update the input shape with the max sequence length. Only update when
      # 1. Input feature is 2D ragged or sparse tensor.
      # 2. Output shape is not set in the feature config and the max sequence
      #    length is set.
      if (len(input_shape) == 2 and input_shape[-1] != 1 and
          not feature.output_shape and feature.max_sequence_length > 0):
        input_shape_list = input_shape.as_list()
        input_shape_list.insert(
            len(input_shape_list) - 1, feature.max_sequence_length)
        input_shape = TensorShape(input_shape_list)
      if input_shape.rank == 1:
        output_shapes.append(input_shape)
      else:
        output_shapes.append(input_shape[:-1])
    return output_shapes

  @property
  def embedding_tables(
      self
  ) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
    # [FUNCTION DOC]
    # Property giving the embedding table variables keyed by table config, for CPU lookups.
    # Under a TPU strategy it raises a runtime error, except while saving a model, where it
    # returns the first shard of each table's parameters. Otherwise it lazily builds the
    # object and returns a mapping from each table config to its parameter variable (slot
    # variables are not included). Side effect: may trigger build.
    """Returns a dict of embedding tables, keyed by `TableConfig`.

    This property only works when the `TPUEmbedding` object is created under a
    non-TPU strategy. This is intended to be used to for CPU based lookup when
    creating a serving checkpoint.

    Returns:
      A dict of embedding tables, keyed by `TableConfig`.

    Raises:
      RuntimeError: If object was created under a `TPUStrategy`.
    """
    # We don't support returning tables on TPU due to their sharded nature and
    # the fact that when using a TPUStrategy:
    # 1. Variables are stale and are only updated when a checkpoint is made.
    # 2. Updating the variables won't affect the actual tables on the TPU.
    if self._using_tpu:
      if save_context.in_save_context():
        return {table: self._variables[table.name]["parameters"].variables[0]
                for table in self._table_config}
      raise RuntimeError("Unable to retrieve embedding tables when using a TPU "
                         "strategy. If you need access, save your model, "
                         "create this object under a CPU strategy and restore.")

    self._maybe_build(None)

    # Only return the tables and not the slot variables. On CPU this are honest
    # tf.Variables.
    return {table: self._variables[table.name]["parameters"]
            for table in self._table_config}

  def _create_config_proto(
      self
  ) -> tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration:
    # [FUNCTION DOC]
    # HELPER: part of build; produces the configuration used to initialize the engine.
    # Builds a new configuration message: one table descriptor per table (using the number
    # of hosts and the dynamic learning rate index mapping), one feature descriptor per
    # feature (name, table id, input shape taken from the output shape), training mode,
    # number of hosts, number of tensor cores (replicas times cores per replica), default
    # division sharding, the pipelining flag, and SPMD sharding when cores per replica is set.
    # Takes no inputs besides state; returns the message. Table descriptors are filled by
    # table configs.
    """Creates the TPUEmbeddingConfiguration proto.

    This proto is used to initialize the TPU embedding engine.

    Returns:
      A TPUEmbeddingConfiguration proto.
    """

    config_proto = tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration()

    # Map each callable dynamic learning rate to its in index in the list.
    # The learning rate index is the index of the dynamic learning rate for this
    # table (if it exists) in the list we created at initialization. We don't
    # simply create one learning rate index per table as this has extremely bad
    # performance characteristics. The more separate optimization configurations
    # we have, the worse the performance will be.
    learning_rate_index = {r: i for i, r in enumerate(
        self._dynamic_learning_rates)}

    for table in self._table_config:
      table._set_table_descriptor(  # pylint: disable=protected-access
          config_proto.table_descriptor.add(),
          self._strategy.extended.num_hosts,
          learning_rate_index)

    table_to_id = {table: i for i, table in enumerate(self._table_config)}

    # Set feature descriptor field in the config proto.
    for feature, output_shape in zip(
        nest.flatten(self._feature_config), self._output_shapes):
      feature_descriptor = config_proto.feature_descriptor.add()

      if feature.name:
        feature_descriptor.name = feature.name

      feature_descriptor.table_id = table_to_id[feature.table]
      # The input shape of the feature is the actual shape of the input tensor
      # except the last dimension because the last dimension will always be
      # reduced.
      feature_descriptor.input_shape.extend(output_shape.as_list())

    # Always set mode to training, we override the mode during enqueue.
    config_proto.mode = (
        tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.TRAINING)

    num_replica = self._strategy.num_replicas_in_sync
    num_cores_per_replica = self._num_cores_per_replica or 1

    config_proto.num_hosts = self._strategy.extended.num_hosts
    config_proto.num_tensor_cores = num_replica * num_cores_per_replica

    # TODO(bfontain): Allow users to pick MOD for the host sharding.
    config_proto.sharding_strategy = (
        tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.DIV_DEFAULT)
    config_proto.pipeline_execution_with_tensor_core = (
        self._pipeline_execution_with_tensor_core)

    if self._num_cores_per_replica:
      config_proto.spmd_sharding.enabled = True
      config_proto.spmd_sharding.num_cores_per_replica = (
          self._num_cores_per_replica
      )

    return config_proto

  def apply_gradients(self, gradients, name: Optional[Text] = None):
    # [FUNCTION DOC]
    # Sends gradients for the embedding activations to the TPU so tables are updated.
    # Inputs: gradients, nested structure matching the feature config whose entries are
    # tensors or nothing; name, optional op name tag. A missing gradient is replaced with a
    # zero tensor and a warning is logged. Each gradient is reshaped to fix its static shape
    # and is validated against the output shape times cores per replica plus table dimension.
    # Raises runtime error when not under a TPU strategy or not built; value error for
    # non-tensor gradients or wrong shapes; structure mismatches raise from the nest check.
    # Side effect: creates the send-gradients op with the current dynamic learning rates and
    # tags it with the name if given. Returns nothing.
    """Applies the gradient update to the embedding tables.

    If a gradient of `None` is passed in any position of the nested structure,
    then an gradient update with a zero gradient is applied for that feature.
    For optimizers like SGD or Adagrad, this is the same as applying no update
    at all. For lazy Adam and other sparsely applied optimizers with decay,
    ensure you understand the effect of applying a zero gradient.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = (
        strategy.distribute_datasets_from_function(
            dataset_fn=...,
            options=tf.distribute.InputOptions(
                experimental_fetch_to_device=False))
    dataset_iterator = iter(distributed_dataset)

    @tf.function
    def training_step():
      def tpu_step(tpu_features):
        with tf.GradientTape() as tape:
          activations = embedding.dequeue()
          tape.watch(activations)

          loss = ... #  some computation involving activations

        embedding_gradients = tape.gradient(loss, activations)
        embedding.apply_gradients(embedding_gradients)

      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(tpu_features, ))

    training_step()
    ```

    Args:
      gradients: A nested structure of gradients, with structure matching the
        `feature_config` passed to this object.
      name: A name for the underlying op.

    Raises:
      RuntimeError: If called when object wasn't created under a `TPUStrategy`
        or if not built (either by manually calling build or calling enqueue).
      ValueError: If a non-`tf.Tensor` non-`None` gradient is passed in, or a
        `tf.Tensor` of the incorrect shape is passed in. Also if
        the size of any sequence in `gradients` does not match corresponding
        sequence in `feature_config`.
      TypeError: If the type of any sequence in `gradients` does not match
        corresponding sequence in `feature_config`.
    """
    if not self._using_tpu:
      raise RuntimeError("apply_gradients is not valid when TPUEmbedding "
                         "object is not created under a TPUStrategy.")

    if not self._built:
      raise RuntimeError("apply_gradients called on unbuilt TPUEmbedding "
                         "object. Please either call enqueue first or manually "
                         "call the build method.")

    num_cores_per_replica = self._num_cores_per_replica or 1

    nest.assert_same_structure(self._feature_config, gradients)
    updated_gradients = []
    for (path, gradient), feature, output_shape in zip(
        nest.flatten_with_joined_string_paths(gradients),
        nest.flatten(self._feature_config), self._output_shapes):
      full_output_shape = [x * num_cores_per_replica for x in output_shape] + [
          feature.table.dim
      ]
      if gradient is not None and not isinstance(gradient, tensor_lib.Tensor):
        raise ValueError(
            f"found non-tensor type: {type(gradient)} at path {path}.")
      if gradient is not None:
        if gradient.shape != full_output_shape:
          raise ValueError("Found gradient of shape {} at path {}. Expected "
                           "shape {}.".format(gradient.shape, path,
                                              full_output_shape))
      else:
        # No gradient for this feature, since we must give a gradient for all
        # features, pass in a zero tensor here. Note that this is not correct
        # for all optimizers.
        logging.warning(
            "No gradient passed for feature %s, sending zero "
            "gradient. This may not be correct behavior for certain "
            "optimizers like Adam.", path)
        gradient = array_ops.zeros(full_output_shape, dtype=dtypes.float32)
      # Some gradients can be passed with op which shape is not correctly set.
      # This ensures that the shape of the gradient is correctly set.
      updated_gradients.append(
          array_ops.reshape(gradient, shape=gradient.shape))
    op = tpu_ops.send_tpu_embedding_gradients(
        inputs=updated_gradients,
        learning_rates=[
            math_ops.cast(fn(), dtype=dtypes.float32)
            for fn in self._dynamic_learning_rates
        ],
        config=self._config_proto.SerializeToString())

    # Apply the name tag to the op.
    if name is not None:
      _add_key_attr(op, name)

  def dequeue(self, name: Optional[Text] = None):
    # [FUNCTION DOC]
    # Receives the embedding lookup results from the TPU. Input: optional op name tag.
    # Raises a runtime error when not created under a TPU strategy or not yet built.
    # Creates a receive-activations op with one output per feature descriptor, tags the first
    # output's op with the name if given, and returns the activations packed into the same
    # nested structure as the feature config.
    """Get the embedding results.

    Returns a nested structure of `tf.Tensor` objects, matching the structure of
    the `feature_config` argument to the `TPUEmbedding` class. The output shape
    of the tensors is `(*output_shape, dim)`, `dim` is the dimension of the
    corresponding `TableConfig`. For output_shape, there are three places where
    it can be set.
      1. FeatureConfig provided in the __init__ function.
      2. Per_replica_output_shapes by directly calling the build method
           after initializing the tpu embedding class.
      3. Auto detected from the shapes of the input feature.
    The priority of these places is the exact same order.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = (
        strategy.distribute_datasets_from_function(
            dataset_fn=...,
            options=tf.distribute.InputOptions(
                experimental_fetch_to_device=False))
    dataset_iterator = iter(distributed_dataset)

    @tf.function
    def training_step():
      def tpu_step(tpu_features):
        with tf.GradientTape() as tape:
          activations = embedding.dequeue()
          tape.watch(activations)

          loss = ... #  some computation involving activations

        embedding_gradients = tape.gradient(loss, activations)
        embedding.apply_gradients(embedding_gradients)

      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(tpu_features, ))

    training_step()
    ```

    Args:
      name: A name for the underlying op.

    Returns:
      A nested structure of tensors, with the same structure as `feature_config`
    passed to this instance of the `TPUEmbedding` object.

    Raises:
      RuntimeError: If called when object wasn't created under a `TPUStrategy`
        or if not built (either by manually calling build or calling enqueue).
    """
    if not self._using_tpu:
      raise RuntimeError("dequeue is not valid when TPUEmbedding object is not "
                         "created under a TPUStrategy.")

    if not self._built:
      raise RuntimeError("dequeue called on unbuilt TPUEmbedding object. "
                         "Please either call enqueue first or manually call "
                         "the build method.")

    # The activations returned by this op are per feature.
    activations = tpu_ops.recv_tpu_embedding_activations(
        num_outputs=len(self._config_proto.feature_descriptor),
        config=self._config_proto.SerializeToString())

    # Apply the name tag to the op.
    if name is not None:
      _add_key_attr(activations[0].op, name)

    # Pack the list back into the same nested structure as the features.
    return nest.pack_sequence_as(self._feature_config, activations)

  def _create_variables_and_slots(
      self
  ) -> Dict[Text, Dict[Text, tf_variables.Variable]]:
    # [FUNCTION DOC]
    # HELPER: part of build; creates parameter and optimizer slot variables for every table.
    # Returns a mapping from table name to a mapping from "parameters" and each slot name to
    # the created variable. Under a TPU strategy creation happens inside a variable creator
    # scope that shards variables across hosts (non-trainable on TPU); otherwise ordinary
    # trainable variables are created. Variables are registered through the trackable
    # custom getter mechanism so checkpoints can restore before creation.
    """Create variables for TPU embeddings.

    Note under TPUStrategy this will ensure that all creations happen within a
    variable creation scope of the sharded variable creator.

    Returns:
      A dict of dicts. The outer dict is keyed by the table names and the inner
      dicts are keyed by 'parameters' and the slot variable names.
    """

    def create_variables(table):
      # [FUNCTION DOC]
      # HELPER: creates the parameter variable and all slot variables for one table.
      # Input: a table config. Returns a mapping from "parameters" and slot names to variables
      # (slots come from the table's optimizer, none if it has no optimizer).
      """Create all variables."""
      variable_shape = (table.vocabulary_size, table.dim)

      def getter(name, shape, dtype, initializer, trainable):
        # [FUNCTION DOC]
        # HELPER: custom getter creating one variable of the full table shape.
        # Inputs: name, ignored shape (the outer table shape is used instead because the
        # trackable machinery may clear it), dtype, initializer and trainable flag.
        # Returns a new variable initialized lazily by the initializer.
        del shape
        # _add_variable_with_custom_getter clears the shape sometimes, so we
        # take the global shape from outside the getter.
        initial_value = functools.partial(initializer, variable_shape,
                                          dtype=dtype)
        return tf_variables.Variable(
            name=name,
            initial_value=initial_value,
            shape=variable_shape,
            dtype=dtype,
            trainable=trainable)

      def variable_creator(name, initializer, trainable=True):
        # [FUNCTION DOC]
        # HELPER: registers a float32 variable on the TPUEmbedding object via the trackable
        # custom getter so it is tracked and can be restored before creation. Inputs: name,
        # initializer, trainable flag (default true). Returns the variable.
        # use add_variable_with_custom_getter here so that we take advantage of
        # the checkpoint loading to allow restore before the variables get
        # created which avoids double initialization.
        return self._add_variable_with_custom_getter(
            name=name,
            initializer=initializer,
            shape=variable_shape,
            dtype=dtypes.float32,
            getter=getter,
            trainable=trainable)

      parameters = variable_creator(table.name, table.initializer,
                                    trainable=not self._using_tpu)

      def slot_creator(name, initializer):
        # [FUNCTION DOC]
        # HELPER: callback handed to the optimizer to create a non-trainable slot variable
        # named after the table and slot name. Inputs: slot name, initializer. Returns it.
        return variable_creator(table.name + "/" + name,
                                initializer,
                                False)

      if table.optimizer is not None:
        slot_vars = table.optimizer._create_slots(parameters, slot_creator)  # pylint: disable=protected-access
      else:
        slot_vars = {}
      slot_vars["parameters"] = parameters
      return slot_vars

    # Store tables based on name rather than TableConfig as we can't track
    # through dicts with non-string keys, i.e. we won't be able to save.
    variables = {}
    for table in self._table_config:
      if not self._using_tpu:
        variables[table.name] = create_variables(table)
      else:
        with variable_scope.variable_creator_scope(
            make_sharded_variable_creator(self._hosts)):
          variables[table.name] = create_variables(table)

    return variables

  def _load_variables(self):
    # [FUNCTION DOC]
    # HELPER: pushes the host variable values onto the TPU (used by build and restore).
    # Only acts when using TPU, built, and not in graph-mode save context; then calls the
    # module level load implementation with the serialized config, hosts, variables and
    # tables. Returns nothing.
    # Only load the variables if we are:
    # 1) Using TPU
    # 2) Variables are created
    # 3) Not in save context (except if running eagerly)
    if self._using_tpu and self._built and not (
        not context.executing_eagerly() and save_context.in_save_context()):
      _load_variables_impl(self._config_proto.SerializeToString(),
                           self._hosts,
                           self._variables,
                           self._table_config)

  def _retrieve_variables(self):
    # [FUNCTION DOC]
    # HELPER: copies table values from the TPU back into host variables (used when saving).
    # Same conditions as loading: TPU in use, built, and not in graph-mode save context.
    # Calls the module level retrieve implementation. Returns nothing.
    # Only retrieve the variables if we are:
    # 1) Using TPU
    # 2) Variables are created
    # 3) Not in save context (except if running eagerly)
    if self._using_tpu and self._built and not (
        not context.executing_eagerly() and save_context.in_save_context()):
      _retrieve_variables_impl(self._config_proto.SerializeToString(),
                               self._hosts,
                               self._variables,
                               self._table_config)

  # Some helper functions for the below enqueue function.
  def _add_data_for_tensor(self, tensor, weight, indices, values, weights,
                           int_zeros, float_zeros, path):
    # [FUNCTION DOC]
    # HELPER: appends dense tensor data for the enqueue op. Inputs: the dense tensor, its
    # weight (must be nothing), and the parallel output lists for indices, values and weights,
    # empty int and float placeholder tensors, and the feature path for messages.
    # Appends placeholder indices, the flattened values cast to 64-bit integers and a
    # placeholder weight. Raises a value error if a weight is supplied. Mutates the lists.
    if weight is not None:
      raise ValueError(
          "Weight specified for dense input {}, which is not allowed. "
          "Weight will always be 1 in this case.".format(path))
    # For tensors, there are no indices and no weights.
    indices.append(int_zeros)
    values.append(math_ops.cast(array_ops.reshape(tensor, [-1]), dtypes.int64))
    weights.append(float_zeros)

  def _add_data_for_sparse_tensor(self, tensor, weight, indices, values,
                                  weights, int_zeros, float_zeros, path,
                                  feature):
    # [FUNCTION DOC]
    # HELPER: appends sparse tensor data for the enqueue op. Inputs as for the dense variant
    # plus the feature config. Appends the indices (cast to 32-bit integers; for rank 2 inputs
    # with a max sequence length and no output shape an extra zero column is added), values
    # cast to 64-bit integers, and weight values cast to float32 or a placeholder. Logs a
    # warning if rank is above 2 and a max sequence length is set (ignored). Raises a value
    # error if a weight is given that is not a sparse tensor. Mutates the lists.
    sample_indices = math_ops.cast(tensor.indices, dtypes.int32)
    if tensor.shape.rank == 2:
      if not feature.output_shape and feature.max_sequence_length > 0:
        # Add one dimension to the last axis.
        sample_indices = array_ops.pad(
            sample_indices, paddings=[[0, 0], [0, 1]])
    else:
      if feature.max_sequence_length > 0:
        logging.warning(
            (
                "Input tensor is rank %d which is above 2, the"
                " max_sequence_length setting will be ignored."
            ),
            tensor.shape.rank,
        )
    indices.append(sample_indices)
    values.append(math_ops.cast(tensor.values, dtypes.int64))
    # If we have weights they must be a SparseTensor.
    if weight is not None:
      if not isinstance(weight, sparse_tensor.SparseTensor):
        raise ValueError("Weight for {} is type {} which does not match "
                         "type input which is SparseTensor.".format(
                             path, type(weight)))
      weights.append(math_ops.cast(weight.values, dtypes.float32))
    else:
      weights.append(float_zeros)

  def _add_data_for_ragged_tensor(self, tensor, weight, row_splits, values,
                                  weights, int_zeros, float_zeros, path,
                                  feature):
    # [FUNCTION DOC]
    # HELPER: appends ragged tensor data for the enqueue op. Appends the row splits (32-bit),
    # values (64-bit) and the weight values (float32) or a placeholder. Raises a value error
    # if a weight is given that is not a ragged tensor. The feature and integer placeholder
    # arguments are unused. Mutates the given lists; returns nothing.
    row_splits.append(math_ops.cast(tensor.row_splits, dtypes.int32))
    values.append(math_ops.cast(tensor.values, dtypes.int64))
    # If we have weights they must be a RaggedTensor.
    if weight is not None:
      if not isinstance(weight, ragged_tensor.RaggedTensor):
        raise ValueError("Weight for {} is type {} which does not match "
                         "type input which is RaggedTensor.".format(
                             path, type(weight)))
      weights.append(math_ops.cast(weight.values, dtypes.float32))
    else:
      weights.append(float_zeros)

  def _generate_enqueue_op(
      self,
      flat_inputs: List[internal_types.NativeObject],
      flat_weights: List[Optional[internal_types.NativeObject]],
      flat_features: List[tpu_embedding_v2_utils.FeatureConfig],
      device_ordinal: int,
      mode_override: Text
  ) -> ops.Operation:
    # [FUNCTION DOC]
    # HELPER: builds the single enqueue op for a batch. Inputs: flat lists of inputs, weights
    # (entries may be nothing) and (path, feature config) pairs, all of equal length; the
    # device ordinal to enqueue for; and the mode override (train or inference string/tensor).
    # Dispatches each input by type (dense, sparse, ragged) to fill parallel lists of indices
    # or row splits, values and weights, with empty placeholders where no data applies;
    # raises a value error for any other input type. Uses per table combiners in table order.
    # Returns the arbitrary-tensor-batch enqueue operation.
    # KNOWN ISSUE: the docstring says it outputs "a the enqueue op"; also the flat_features
    # entries are actually (path, config) pairs, not plain FeatureConfigs as the type hint says.
    """Outputs a the enqueue op given the inputs and weights.

    Args:
      flat_inputs: A list of input tensors.
      flat_weights: A list of input weights (or None) of the same length as
        flat_inputs.
      flat_features: A list of FeatureConfigs of the same length as flat_inputs.
      device_ordinal: The device to create the enqueue op for.
      mode_override: A tensor containing the string "train" or "inference".

    Returns:
      The enqueue op.
    """
    # Combiners are per table, list in the same order as the table order.
    combiners = [table.combiner for table in self._table_config]

    # These parallel arrays will be the inputs to the enqueue op.
    # sample_indices for sparse, row_splits for ragged.
    indices_or_row_splits = []
    values = []
    weights = []

    # We have to supply a empty/zero tensor in a list position where we don't
    # have data (e.g. indices for standard Tensor input, weight when no weight
    # is specified). We create one op here per call, so that we reduce the
    # graph size.
    int_zeros = array_ops.zeros((0,), dtype=dtypes.int32)
    float_zeros = array_ops.zeros((0,), dtype=dtypes.float32)

    # In the following loop we insert casts so that everything is either int32
    # or float32. This is because op inputs which are lists of tensors must be
    # of the same type within the list. Moreover the CPU implementations of
    # these ops cast to these types anyway, so we don't lose any data by casting
    # early.
    for inp, weight, (path, feature) in zip(
        flat_inputs, flat_weights, flat_features):
      if isinstance(inp, tensor_lib.Tensor):
        self._add_data_for_tensor(inp, weight, indices_or_row_splits, values,
                                  weights, int_zeros, float_zeros, path)
      elif isinstance(inp, sparse_tensor.SparseTensor):
        self._add_data_for_sparse_tensor(inp, weight, indices_or_row_splits,
                                         values, weights, int_zeros,
                                         float_zeros, path, feature)
      elif isinstance(inp, ragged_tensor.RaggedTensor):
        self._add_data_for_ragged_tensor(inp, weight, indices_or_row_splits,
                                         values, weights, int_zeros,
                                         float_zeros, path, feature)
      else:
        raise ValueError("Input {} is of unknown type {}. Please only pass "
                         "Tensor, SparseTensor or RaggedTensor as input to "
                         "enqueue.".format(path, type(inp)))

    return tpu_ops.enqueue_tpu_embedding_arbitrary_tensor_batch(
        sample_indices_or_row_splits=indices_or_row_splits,
        embedding_indices=values,
        aggregation_weights=weights,
        mode_override=mode_override,
        device_ordinal=device_ordinal,
        combiners=combiners)

  def _raise_error_for_incorrect_control_flow_context(self):
    # [FUNCTION DOC]
    # HELPER: validates where enqueue is called from. Walks the current graph and its outer
    # graphs looking for a TPU replicate control flow context. Returns true if one is found,
    # false otherwise. Raises a runtime error if the context belongs to a different graph than
    # the current one (enqueue inside control flow or a nested function within a strategy run).
    # KNOWN ISSUE: the docstring says it raises when not in a TPU replicate context, but it
    # only raises for a graph mismatch and otherwise returns a boolean.
    """Raises an error if we are not in the TPUReplicateContext."""
    # Do not allow any XLA control flow (i.e. control flow in between a
    # TPUStrategy's run call and the call to this function), as we can't
    # extract the enqueue from the head when in XLA control flow.
    graph = ops.get_default_graph()
    in_tpu_ctx = False
    while graph is not None:
      ctx = graph._get_control_flow_context()  # pylint: disable=protected-access
      while ctx is not None:
        if isinstance(ctx, tpu_replication.TPUReplicateContext):
          in_tpu_ctx = True
          break
        ctx = ctx.outer_context
      if in_tpu_ctx:
        break
      graph = getattr(graph, "outer_graph", None)
    if graph != ops.get_default_graph() and in_tpu_ctx:
      raise RuntimeError(
          "Current graph {} does not match graph which contains "
          "TPUReplicateContext {}. This is most likely due to the fact that "
          "enqueueing embedding data is called inside control flow or a "
          "tf.function inside `strategy.run`. This is not supported because "
          "outside compilation fails to extract the enqueue ops as the head of "
          "a computation.".format(ops.get_default_graph(), graph))
    return in_tpu_ctx

  def _raise_error_for_non_direct_inputs(self, features):
    # [FUNCTION DOC]
    # HELPER: ensures inputs inside a strategy run come directly from the function arguments.
    # Input: nested structure of features. Every component tensor (composites expanded) must
    # be a placeholder or the output of an op carrying the TPU input identity attribute;
    # otherwise raises a value error describing likely causes (dynamic batch size, keras
    # dtype mismatch). Returns nothing.
    """Checks all tensors in features to see if they are a direct input."""

    # expand_composites here is important: as composite tensors pass through
    # tpu.replicate, they get 'flattened' into their component tensors and then
    # repacked before being passed to the tpu function. In means that it is the
    # component tensors which are produced by an op with the
    # "_tpu_input_identity" attribute.
    for path, input_tensor in nest.flatten_with_joined_string_paths(
        features, expand_composites=True):
      if input_tensor.op.type == "Placeholder":
        continue
      try:
        is_input = input_tensor.op.get_attr("_tpu_input_identity")
      except ValueError:
        is_input = False
      if not is_input:
        raise ValueError(
            "Received input tensor {} which is the output of op {} (type {}) "
            "which does not have the `_tpu_input_identity` attr. Please "
            "ensure that the inputs to this layer are taken directly from "
            "the arguments of the function called by "
            "strategy.run. Two possible causes are: dynamic batch size "
            "support or you are using a keras layer and are not passing "
            "tensors which match the dtype of the `tf.keras.Input`s."
            "If you are triggering dynamic batch size support, you can "
            "disable it by passing tf.distribute.RunOptions("
            "experimental_enable_dynamic_batch_size=False) to the options "
            "argument of strategy.run().".format(path,
                                                 input_tensor.op.name,
                                                 input_tensor.op.type))

  def _raise_error_for_inputs_not_on_cpu(self, flat_inputs, flat_paths):
    # [FUNCTION DOC]
    # HELPER: verifies that enqueue inputs live on the host. Inputs: flat list of input
    # tensors (possibly composite) and the parallel list of feature paths. Checks the device
    # of every underlying tensor (looking through replicated-input identities to each
    # replica's device) and raises a value error if any is on a TPU. Returns nothing.
    """Checks all tensors in features to see are placed on the CPU."""

    def check_device(path, device_string):
      # [FUNCTION DOC]
      # HELPER: raises a value error naming the feature path if the device string denotes a
      # TPU device. Inputs: feature path and device string. Returns nothing.
      spec = tf_device.DeviceSpec.from_string(device_string)
      if spec.device_type == "TPU":
        raise ValueError(
            "Received input tensor {} which is on a TPU input device {}. Input "
            "tensors for TPU embeddings must be placed on the CPU. Please "
            "ensure that your dataset is prefetching tensors to the host by "
            "setting the 'experimental_fetch_to_device' option of the "
            "dataset distribution function. See the documentation of the "
            "enqueue method for an example.".format(path, device_string))

    # expand_composites here is important, we need to check the device of each
    # underlying tensor.
    for input_tensor, input_path in zip(flat_inputs, flat_paths):
      if nest.is_nested_or_composite(input_tensor):
        input_tensors = nest.flatten(input_tensor, expand_composites=True)
      else:
        input_tensors = [input_tensor]
      for t in input_tensors:
        if (t.op.type == "Identity" and
            t.op.inputs[0].op.type == "TPUReplicatedInput"):
          for tensor in t.op.inputs[0].op.inputs:
            check_device(input_path, tensor.device)
        else:
          check_device(input_path, t.device)

  def enqueue(
      self,
      features,
      weights=None,
      training: bool = True,
      name: Optional[Text] = None,
      device: Optional[Text] = None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says inputs are downcast to int32, but dense and sparse ids
    # are cast to 64-bit integers before the enqueue op (row splits are 32-bit). The Args
    # entries for name and device are also indented wrongly, and the finer grained example
    # refers to a non-existent per-core function name. Also the list of enqueue ops built in
    # the per-replica branch is collected but never returned or used.
    # Enqueues a batch of ids (and optional weights) for lookup. Inputs: features, nested
    # structure of dense, sparse or ragged tensors matching the feature config; weights,
    # optional matching structure of float tensors; training, bool default true selecting
    # train or inference mode; name, optional op name tag; device, optional TPU device name
    # used when not in a TPU context and features are not per replica values.
    # Raises runtime error when not under a TPU strategy or in XLA control flow, value error
    # for non-direct inputs, inputs on TPU, shape mismatches, unknown types, ragged inputs
    # with SPMD splitting, or a non-TPU device. Lazily builds the object from the detected
    # input shapes, validates them against stored output shapes. In a TPU context it adds
    # the enqueue via outside compilation; with no device it creates an enqueue op per
    # replica and core on the matching host (splitting inputs when cores per replica is set);
    # with a device it creates one op on that device's host. Returns nothing.
    """Enqueues id tensors for embedding lookup.

    This function enqueues a structure of features to be looked up in the
    embedding tables. We expect that the input shapes of each of the tensors in
    features matches the output shapes set via FeatureConfig or build method
    (if any). the output shapes will be auto detected based on the input shapes
    with the max_sequence_length or output shape setting in the FeatureConfig.
    Note that the output shapes is based on per replica batch size.
    If your input dataset is batched to the global batch size and you use
    `tf.distribute.TPUStrategy`'s `experimental_distribute_dataset`
    or if you use `distribute_datasets_from_function` and batch
    to the per core batch size computed by the context passed to your input
    function, the output shapes should match automatically.

    The auto detected the output shapes:
      1. For dense tensor, if rank 2 or above, make sure the tensor has last
         dimension as 1. The output shape will be the input shape excluding
         the last dimension.
      2. For sparse tensor, make sure the tensor has rank 2 and above.
           a. If feature config has max_sequence_length equals 0 or output shape
              set (the max_sequence_length setting will be ignored), the
              output shape will be the input shape excluding the last dimension.
           b. Otherwise, if the tensor is rank 2, the output shape will be input
              shape  with last dimension set as max_sequence_length. If the
              tensor is above rank 2, the output shape will be the input shape
              excluding the last dimension and the last dimension of the output
              shape will be set to max_sequence_length.
      3. For ragged tensor, make sure the tensor has rank 2.
           a. If feature config has max_sequence_length equals 0 or output shape
              set (the max_sequence_length setting will be ignored), the
              output shape will be the input shape excluding the last dimension.
           b. Otherwise, the output shape will be the input shape excluding the
              last dimension and the last dimension of the output shape will be
              set to max_sequence_length.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = (
        strategy.distribute_datasets_from_function(
            dataset_fn=...,
            options=tf.distribute.InputOptions(
                experimental_fetch_to_device=False))
    dataset_iterator = iter(distributed_dataset)

    @tf.function
    def training_step():
      def tpu_step(tpu_features):
        with tf.GradientTape() as tape:
          activations = embedding.dequeue()
          tape.watch(activations)

          loss = ... #  some computation involving activations

        embedding_gradients = tape.gradient(loss, activations)
        embedding.apply_gradients(embedding_gradients)

      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(tpu_features,))

    training_step()
    ```

    NOTE: You should specify `training=True` when using
    `embedding.apply_gradients` as above and `training=False` when not using
    `embedding.apply_gradients` (e.g. for frozen embeddings or when doing
    evaluation).

    For finer grained control, in the above example the line

    ```
      embedding.enqueue(embedding_features, training=True)
    ```

    may be replaced with

    ```
      per_core_embedding_features = self.strategy.experimental_local_results(
          embedding_features)

      def per_core_enqueue(ctx):
        core_id = ctx.replica_id_in_sync_group
        device = strategy.extended.worker_devices[core_id]
        embedding.enqueue(per_core_embedding_features[core_id],
                          device=device)

      strategy.experimental_distribute_values_from_function(
          per_core_queue_inputs)
    ```

    Args:
      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or
        `tf.RaggedTensor`s, with the same structure as `feature_config`. Inputs
        will be downcast to `tf.int32`. Only one type out of `tf.SparseTensor`
        or `tf.RaggedTensor` is supported per call.
      weights: If not `None`, a nested structure of `tf.Tensor`s,
        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except
        that the tensors should be of float type (and they will be downcast to
        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the
        same for the parallel entries from `features` and similarly for
        `tf.RaggedTensor`s we assume the row_splits are the same.
      training: Defaults to `True`. If `False`, enqueue the batch as inference
        batch (forward pass only). Do not call `apply_gradients` when this is
        `False` as this may lead to a deadlock.
       name: A name for the underlying op.
       device: The device name (e.g. '/task:0/device:TPU:2') where this batch
         should be enqueued. This should be set if and only if features is not a
         `tf.distribute.DistributedValues` and enqueue is not being called
         inside a TPU context (e.g. inside `TPUStrategy.run`).

    Raises:
      ValueError: When called inside a strategy.run call and input is not
        directly taken from the args of the `strategy.run` call. Also if
        the size of any sequence in `features` does not match corresponding
        sequence in `feature_config`. Similarly for `weights`, if not `None`.
        If input shapes of features is unequal or different from a previous
        call.
      RuntimeError: When called inside a strategy.run call and inside XLA
        control flow. If batch_size is not able to be determined and build was
        not called.
      TypeError: If the type of any sequence in `features` does not match
        corresponding sequence in `feature_config`. Similarly for `weights`, if
        not `None`.
    """
    if not self._using_tpu:
      raise RuntimeError("enqueue is not valid when TPUEmbedding object is not "
                         "created under a TPUStrategy.")

    in_tpu_context = self._raise_error_for_incorrect_control_flow_context()

    nest.assert_same_structure(self._feature_config, features)

    if not self._verify_output_shapes_on_enqueue:
      if not self._output_shapes or not self._built:
        raise ValueError(
            "Configured not to check output shapes on each enqueue() call; please "
            "ensure build() was called with output shapes to initialize "
            "the TPU for embeddings.")
    else:
      per_replica = device is None
      input_shapes = self._get_input_shapes(
          features, per_replica, in_tpu_context
      )

      self._maybe_build(input_shapes)
      # If is already built, we still need to check if the output shapes matches
      # with the previous ones.
      self._check_output_shapes(
          self._get_output_shapes_from_input_shapes(input_shapes))

    flat_inputs = nest.flatten(features)
    flat_weights = [None] * len(flat_inputs)
    if weights is not None:
      nest.assert_same_structure(self._feature_config, weights)
      flat_weights = nest.flatten(weights)
    flat_features = nest.flatten_with_joined_string_paths(self._feature_config)
    flat_paths, _ = zip(*flat_features)

    self._raise_error_for_inputs_not_on_cpu(flat_inputs, flat_paths)
    # If we are in a tpu_context, automatically apply outside compilation.
    if in_tpu_context:
      self._raise_error_for_non_direct_inputs(features)

      def generate_enqueue_ops():
        # [FUNCTION DOC]
        # Nested helper of enqueue run through outside compilation. Builds the train or
        # inference mode selection value from the training flag, generates the enqueue op
        # with device ordinal -1 (fixed by a later rewrite) and tags it with the name if
        # given. No inputs; returns nothing.
        """Generate enqueue ops for outside compilation."""
        # Note that we put array_ops.where_v2 rather than a python if so that
        # the op is explicitly create and the constant ops are both in the graph
        # even though we don't expect training to be a tensor (and thus generate
        # control flow automatically). This need to make it easier to re-write
        # the graph later if we need to fix which mode needs to be used.
        mode_override = array_ops.where_v2(training,
                                           constant_op.constant("train"),
                                           constant_op.constant("inference"))
        # Device ordinal is -1 here, a later rewrite will fix this once the op
        # is expanded by outside compilation.
        enqueue_op = self._generate_enqueue_op(
            flat_inputs, flat_weights, flat_features, device_ordinal=-1,
            mode_override=mode_override)

        # Apply the name tag to the op.
        if name is not None:
          _add_key_attr(enqueue_op, name)

      tpu_replication.outside_compilation(generate_enqueue_ops)

    elif device is None:
      mode_override = "train" if training else "inference"
      # We generate enqueue ops per device, so we need to gather the all
      # features for a single device in to a dict.
      # We rely here on the fact that the devices in the PerReplica value occur
      # in the same (standard) order as self._strategy.extended.worker_devices.
      enqueue_ops = []

      def _split_fn(ts, idx):
        # [FUNCTION DOC]
        # HELPER: takes the idx-th slice of an input along the batch axis when splitting for
        # SPMD cores. Inputs: a tensor, sparse tensor or nothing, and the core index. Returns
        # nothing for nothing, the slice for dense or sparse inputs, and raises a value error
        # for ragged tensors (unsupported).
        if ts is None:
          return None
        elif isinstance(ts, tensor_lib.Tensor):
          return array_ops.split(
              ts,
              num_or_size_splits=self._num_cores_per_replica,
              axis=0)[idx]
        elif isinstance(ts, sparse_tensor.SparseTensor):
          return sparse_ops.sparse_split_v2(
              sp_input=ts,
              num_split=self._num_cores_per_replica,
              axis=0)[idx]
        else:
          raise ValueError("SPMD does not support raggedTensor yet.")

      def _maybe_split(ts_inputs, core_id):
        # [FUNCTION DOC]
        # HELPER: returns the inputs unchanged when cores per replica is not set, otherwise
        # maps the split function over the nested inputs to get the slice for core_id.
        if self._num_cores_per_replica is None:
          return ts_inputs
        else:
          splitter = functools.partial(_split_fn, idx=core_id)
          return nest.map_structure(splitter, ts_inputs)

      for replica_id in range(self._strategy.num_replicas_in_sync):
        replica_inputs = distribute_utils.select_replica(replica_id,
                                                         flat_inputs)
        replica_weights = distribute_utils.select_replica(replica_id,
                                                          flat_weights)

        if self._num_cores_per_replica:
          tpu_devices = self._strategy.extended._tpu_devices[replica_id]   # pylint: disable=protected-access
        else:
          tpu_devices = [self._strategy.extended.worker_devices[replica_id]]
        # TPU devices string are like /job:worker/replica:0/task:0/device:TPU:0
        # the device ordinal is the last number

        for core_id in range(self._num_cores_per_replica or 1):
          tpu_device = tpu_devices[core_id]
          device_ordinal = (
              tf_device.DeviceSpec.from_string(tpu_device).device_index)

          with ops.device(device_util.get_host_for_device(tpu_device)):
            enqueue_op = self._generate_enqueue_op(
                _maybe_split(replica_inputs, core_id),
                _maybe_split(replica_weights, core_id),
                flat_features,
                device_ordinal=device_ordinal, mode_override=mode_override)

            # Apply the name tag to the op.
            if name is not None:
              _add_key_attr(enqueue_op, name)
            enqueue_ops.append(enqueue_op)
    else:
      mode_override = "train" if training else "inference"
      device_spec = tf_device.DeviceSpec.from_string(device)
      if device_spec.device_type != "TPU":
        raise ValueError(
            "Non-TPU device {} passed to enqueue.".format(device))

      with ops.device(device_util.get_host_for_device(device)):
        enqueue_op = self._generate_enqueue_op(
            flat_inputs, flat_weights, flat_features,
            device_ordinal=device_spec.device_index,
            mode_override=mode_override)

        # Apply the name tag to the op.
        if name is not None:
          _add_key_attr(enqueue_op, name)

  def _get_input_shapes(
      self, tensors, per_replica: bool, in_tpu_context: bool
  ) -> List[TensorShape]:
    # [FUNCTION DOC]
    # HELPER: computes the flat list of input shapes for enqueue. Inputs: nested structure
    # of features, whether shapes are per replica, and whether in a TPU context (otherwise
    # the first replica's value is used). Dispatches by tensor type to the shape helpers.
    # KNOWN ISSUE: a feature of an unrecognized type is silently skipped, so the returned list
    # can be shorter than the feature list.
    """Get the input shapes from the input tensor."""
    input_shapes = []
    for (path, maybe_tensor), feature in zip(
        nest.flatten_with_joined_string_paths(tensors),
        nest.flatten(self._feature_config)):
      if not in_tpu_context:
        tensor = distribute_utils.select_replica(0, maybe_tensor)
      else:
        tensor = maybe_tensor

      if isinstance(tensor, tensor_lib.Tensor):
        input_shapes.append(
            self._get_input_shape_for_tensor(tensor, feature, per_replica, path)
        )
      elif isinstance(tensor, sparse_tensor.SparseTensor):
        input_shapes.append(
            self._get_input_shape_for_sparse_tensor(
                tensor, feature, per_replica, path
            )
        )
      elif isinstance(tensor, ragged_tensor.RaggedTensor):
        input_shapes.append(
            self._get_input_shape_for_ragged_tensor(
                tensor, feature, per_replica, path
            )
        )
    return input_shapes

  def _get_input_shape_for_tensor(
      self, tensor, feature, per_replica, path
  ) -> TensorShape:
    # [FUNCTION DOC]
    # HELPER: input shape of a dense feature. Inputs: tensor, feature config (unused), per
    # replica flag and feature path. Requires rank one or more and, for rank two and above,
    # a last dimension of one; otherwise raises a value error. When cores per replica is set
    # and per replica is requested, divides the batch dimension by it. Returns the shape.
    # KNOWN ISSUE: the rank error message wrongly says "sparse tensor" for a dense tensor.
    """Get the input shape for the dense tensor."""
    shape = tensor.shape.as_list()
    if len(shape) < 1:
      raise ValueError("Only rank 1 and above dense tensor is supported,"
                       " find rank {} sparse tensor for input {}".format(
                           len(shape), path))
    if len(shape) > 1 and shape[-1] != 1:
      raise ValueError(
          "Rank 2 or above dense tensor should have last dimension as 1 "
          "as the last dimension will always be reduced. "
          "Instead got dense tensor as shape {}".format(shape))

    if self._num_cores_per_replica and per_replica:
      shape[0] = shape[0] // self._num_cores_per_replica

    return TensorShape(shape)

  def _get_input_shape_for_sparse_tensor(
      self, tensor, feature, per_replica, path
  ) -> TensorShape:
    # [FUNCTION DOC]
    # HELPER: input shape of a sparse feature. Inputs: tensor, feature config, per replica
    # flag, feature path. Requires rank two or more (value error otherwise). For rank two with
    # no configured output shape and a positive max sequence length, inserts the sequence
    # length before the last dimension. Divides a known batch dimension by cores per replica
    # when per replica applies. Returns the shape.
    """Get the input shape for the sparse tensor."""
    shape = tensor.shape.as_list()
    # Only 2 and above rank sparse tensor is supported.
    if len(shape) < 2:
      raise ValueError("Only rank 2 and above sparse tensor is supported,"
                       " find rank {} sparse tensor for input {}".format(
                           len(shape), path))
    if not feature.output_shape and feature.max_sequence_length > 0:
      # If the max_sequence_length is set and the output shape for FeatureConfig
      # is not set, we modify the shape of the input feature. Only rank 2
      # feature output shape is modified
      if len(shape) == 2:
        # If the sparse tensor is 2D and max_sequence_length is set,
        # we need to add one dimension to the input feature.
        shape.insert(len(shape) - 1, feature.max_sequence_length)

    if self._num_cores_per_replica and per_replica and shape[0]:
      shape[0] = shape[0] // self._num_cores_per_replica

    return TensorShape(shape)

  def _get_input_shape_for_ragged_tensor(
      self, tensor, feature, per_replica, path
  ) -> TensorShape:
    # [FUNCTION DOC]
    # HELPER: input shape of a ragged feature. Inputs: tensor, feature config, per replica
    # flag (ignored), feature path. Requires rank two (value error otherwise). With no
    # configured output shape and a positive max sequence length, inserts the sequence length
    # before the last dimension. Returns the shape.
    """Get the input shape for the ragged tensor."""
    del per_replica  # unused.
    shape = tensor.shape.as_list()
    # Only rank 2 ragged tensor is supported.
    if len(shape) != 2:
      raise ValueError("Only rank 2 ragged tensor is supported,"
                       " find rank {} ragged tensor for input {}".format(
                           len(shape), path))
    if not feature.output_shape and feature.max_sequence_length > 0:
      # If the max_sequence_length is set and the output shape for FeatureConfig
      # is not set, add the sequence length as second last dimension of
      # the ragged tensor.
      shape.insert(len(shape) - 1, feature.max_sequence_length)

    return TensorShape(shape)

  def _update_output_shapes(self, incoming_output_shapes: List[TensorShape]):
    # [FUNCTION DOC]
    # HELPER: fills missing stored output shapes from newly detected ones. Input: list of
    # incoming output shapes with the same structure as the stored ones (structure mismatch
    # raises). Existing non-empty shapes win; empty ones are replaced. Replaces the stored
    # list; returns nothing.
    """Update the existing output shapes based on the new output shapes.

    The existing output shapes always have higher piority than the new incoming
    output shapes.
    Args:
      incoming_output_shapes: nested structure of TensorShape to override the
        existing output shapes.
    """
    nest.assert_same_structure(self._output_shapes, incoming_output_shapes)
    updated_output_shapes = []
    for old_output_shape, incoming_output_shape in zip(self._output_shapes,
                                                       incoming_output_shapes):
      if old_output_shape:
        updated_output_shapes.append(old_output_shape)
      else:
        updated_output_shapes.append(incoming_output_shape)
    self._output_shapes = updated_output_shapes

  def _check_output_shapes(self, incoming_output_shapes: List[TensorShape]):
    # [FUNCTION DOC]
    # HELPER: validates incoming output shapes against stored ones. Input: list of incoming
    # shapes (structure mismatch raises). Where both exist, skips the check if the incoming
    # rank is one or two and the stored rank is larger (ragged sequences, checked later in
    # the enqueue op); otherwise raises a value error naming the feature path when ranks
    # differ or known dimensions disagree. Returns nothing.
    """Check the incoming output shapes against the output shapes stored."""
    # The incoming output shape should have the same structure with the existing
    # output shapes.
    nest.assert_same_structure(self._output_shapes, incoming_output_shapes)

    for (path, _), old_output_shape, incoming_output_shape in zip(
        nest.flatten_with_joined_string_paths(self._feature_config),
        self._output_shapes, incoming_output_shapes):
      # First check if both shapes are not None.
      if old_output_shape and incoming_output_shape:
        # We skip the check when the incoming output shape is rank 1 or 2 and
        # rank of the old output shape is larger. This can happen for
        # (sequence) ragged tensor, we push the check down to the enqueue op.
        if (len(incoming_output_shape) == 1 or len(incoming_output_shape)
            == 2) and len(old_output_shape) > len(incoming_output_shape):
          continue
        if len(old_output_shape) != len(
            incoming_output_shape) or not self._is_tensor_shape_match(
                old_output_shape, incoming_output_shape):
          raise ValueError(
              f"Inconsistent shape founded for input feature {path}, "
              f"Output shape is set to be {old_output_shape}, "
              f"But got incoming output shape {incoming_output_shape}")

  def _check_output_shapes_fully_defined(self):
    # [FUNCTION DOC]
    # HELPER: raises a value error naming the feature path if any stored output shape is not
    # fully defined, since shapes are needed for the configuration proto. Returns nothing.
    """Check if the output shape is fully defined."""
    for (path, _), output_shape in zip(
        nest.flatten_with_joined_string_paths(self._feature_config),
        self._output_shapes):
      if not output_shape.is_fully_defined():
        raise ValueError(
            f"Input Feature {path} has output shape set as "
            f"{output_shape} which is not fully defined. "
            "Please specify the fully defined shape in either FeatureConfig "
            "or for the build method.")

  def _is_tensor_shape_match(self, shape_a: TensorShape,
                             shape_b: TensorShape) -> bool:
    # [FUNCTION DOC]
    # HELPER: compares two shapes dimension by dimension, ignoring unknown (empty) dimensions.
    # Returns true if all pairs of known dimensions are equal, false otherwise. Only the
    # common prefix length is compared (extra dimensions are ignored).
    """Check if shape b matches with shape a."""
    for s_a, s_b in zip(shape_a.as_list(), shape_b.as_list()):
      if s_a and s_b and s_a != s_b:
        return False
    return True

  def _get_output_shapes_from_batch_size(self, per_replica_batch_size):
    # [FUNCTION DOC]
    # HELPER: builds output shapes from a legacy per replica batch size. Input: integer batch
    # size. For each feature without an output shape and with a positive max sequence length
    # the shape is batch size by sequence length; otherwise it is just the batch size (rank
    # one). Returns the list of shapes.
    """Get the output shapes from the batch size."""
    output_shapes = []
    for feature in nest.flatten(self._feature_config):
      if not feature.output_shape and feature.max_sequence_length > 0:
        output_shapes.append(
            TensorShape([per_replica_batch_size, feature.max_sequence_length]))
      else:
        output_shapes.append(TensorShape(per_replica_batch_size))
    return output_shapes

  def _create_copy_for_async_checkpoint(
      self, feature_config, optimizer, pipeline_execution_with_tensor_core):
    # [FUNCTION DOC]
    # HELPER: factory used by the async checkpoint helper to make a fresh TPUEmbedding.
    # Inputs: feature config, optimizer and pipelining flag, passed to the constructor.
    # Returns the new object (not built).
    """Create a TPUEmbedding copy for checkpoint/async_checkpoint_helper.py."""
    return TPUEmbedding(
        feature_config=feature_config,
        optimizer=optimizer,
        pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core)


@def_function.function
def _load_variables_impl(
    config: Text,
    hosts: List[Tuple[int, Text]],
    variables: Dict[Text, Dict[Text, tf_variables.Variable]],
    table_config: tpu_embedding_v2_utils.TableConfig):
  # [FUNCTION DOC]
  # HELPER: traced function that loads host variable values into the TPU embedding engine.
  # Inputs: serialized config proto, list of host devices, mapping from table name to
  # mapping of parameter and slot names to sharded variables, and the tables.
  # For each host, selects that host's shard (zeros shaped like the first shard when there
  # are more hosts than shards) and calls each table optimizer's load op with the table name,
  # shard count and id. The config is passed only on the first call to avoid bloating the
  # graph. Returns nothing; side effect is writing table state onto the TPU.
  # KNOWN ISSUE: table_config is annotated as a single TableConfig but is iterated as a list.
  """Load embedding tables to onto TPU for each table and host.

  Args:
    config: A serialized TPUEmbeddingConfiguration proto.
    hosts: A list of CPU devices, on per host.
    variables: A dictionary of dictionaries of TPUEmbeddingVariables. First key
      is the table name, second key is 'parameters' or the optimizer slot name.
    table_config: A list of tf.tpu.experimental.embedding.TableConfig objects.
  """
  def select_fn(host_id):
    # [FUNCTION DOC]
    # HELPER: given a host index, returns a function choosing that host's shard variable.
    # Input: host index. Returns the selector function.

    def select_or_zeros(x):
      # [FUNCTION DOC]
      # HELPER: selector returned by the enclosing function. Input: a sharded variable.
      # Returns its shard for the host, or zeros shaped like the first shard when the host
      # index is beyond the available shards.
      if host_id >= len(x.variables):
        # In the edge case where we have more hosts than variables, due to using
        # a small number of rows, we load zeros for the later hosts. We copy
        # the shape of the first host's variables, which we assume is defined
        # because TableConfig guarantees at least one row.
        return array_ops.zeros_like(x.variables[0])
      return x.variables[host_id]

    return select_or_zeros

  for host_id, host in enumerate(hosts):
    with ops.device(host):
      host_variables = nest.map_structure(select_fn(host_id), variables)
      for table in table_config:
        table.optimizer._load()(  # pylint: disable=protected-access
            table_name=table.name,
            num_shards=len(hosts),
            shard_id=host_id,
            config=config,
            **host_variables[table.name])
        # Ensure that only the first table/first host gets a config so that we
        # don't bloat graph by attaching this large string to each op.
        # We have num tables * num hosts of these so for models with a large
        # number of tables training on a large slice, this can be an issue.
        config = None


@def_function.function
def _retrieve_variables_impl(
    config: Text,
    hosts: List[Tuple[int, Text]],
    variables: Dict[Text, Dict[Text, tf_variables.Variable]],
    table_config: tpu_embedding_v2_utils.TableConfig):
  # [FUNCTION DOC]
  # HELPER: traced function that copies table values from the TPU into host variables.
  # Inputs as for the load counterpart. For each host and table, calls the optimizer's
  # retrieve op, wraps a single result into a tuple, and assigns each returned tensor
  # (parameters first, then slots) to the matching shard variable, skipping hosts without a
  # shard. Config is passed only on the first call. Returns nothing; side effect is assigning
  # the host variables.
  # KNOWN ISSUE: table_config is annotated as a single TableConfig but is iterated as a list.
  """Retrieve embedding tables from TPU to host memory.

  Args:
    config: A serialized TPUEmbeddingConfiguration proto.
    hosts: A list of all the host CPU devices.
    variables: A dictionary of dictionaries of TPUEmbeddingVariables. First key
      is the table name, second key is 'parameters' or the optimizer slot name.
    table_config: A list of tf.tpu.experimental.embedding.TableConfig objects.
  """
  for host_id, host in enumerate(hosts):
    with ops.device(host):
      for table in table_config:
        retrieved = table.optimizer._retrieve()(  # pylint: disable=protected-access
            table_name=table.name,
            num_shards=len(hosts),
            shard_id=host_id,
            config=config)
        # When there are no slot variables (e.g with SGD) this returns a
        # single tensor rather than a tuple. In this case we put the tensor in
        # a list to make the following code easier to write.
        if not isinstance(retrieved, tuple):
          retrieved = (retrieved,)

        for i, slot in enumerate(["parameters"] +
                                 table.optimizer._slot_names()):  # pylint: disable=protected-access
          # We must assign the CPU variables the values of tensors that were
          # returned from the TPU.
          sharded_var = variables[table.name][slot]
          if host_id < len(sharded_var.variables):
            # In the edge case where we have more hosts than variables, due to
            # using a small number of rows, we skip the later hosts.
            sharded_var.variables[host_id].assign(retrieved[i])
        # Ensure that only the first table/first host gets a config so that we
        # don't bloat graph by attaching this large string to each op.
        # We have num tables * num hosts of these so for models with a large
        # number of tables training on a large slice, this can be an issue.
        config = None


def _save_callback(trackables, **unused_kwargs):
  # [FUNCTION DOC]
  # HELPER: checkpoint saver hook. Input: mapping of names to TPUEmbedding objects; extra
  # keyword arguments ignored. Retrieves every object's variables from the TPU so the host
  # copies are current before saving. Returns an empty list (no extra save operations).
  for trackable in trackables.values():
    trackable._retrieve_variables()  # pylint: disable=protected-access
  return []


def _restore_callback(trackables, **unused_kwargs):
  # [FUNCTION DOC]
  # HELPER: checkpoint restore hook. Input: mapping of names to TPUEmbedding objects; extra
  # keyword arguments ignored. Loads each object's restored host variables onto the TPU.
  # Returns nothing.
  for trackable in trackables.values():
    trackable._load_variables()  # pylint: disable=protected-access


registration.register_tf_checkpoint_saver(
    "TPUEmbeddingCallback",
    predicate=lambda x: isinstance(x, TPUEmbedding),
    save_fn=_save_callback,
    restore_fn=_restore_callback,
    # Set strict_predicate_restore to `False` to because the isinstance
    # predicate check does not pass after a TPUEmbedding object is loaded from
    # SavedModel.
    strict_predicate_restore=False
)


def extract_variable_info(
    kwargs) -> Tuple[Text, Tuple[int, ...], dtypes.DType, Callable[[], Any]]:
  # [FUNCTION DOC]
  # HELPER: used by the sharded variable creator to recover how to initialize shards.
  # Input: keyword arguments given to a variable creator (name, dtype, initial value, shape).
  # If the initial value is a partial application with the shape set by keyword or first
  # positional argument, returns the name, that shape, its dtype keyword (falling back to the
  # dtype argument) and the wrapped initializer function. Otherwise, if the shape is missing
  # or the initial value is not callable, raises a value error; else returns name, shape,
  # dtype and initial value as given.
  """Extracts the variable creation attributes from the kwargs.

  Args:
    kwargs: a dict of keyword arguments that were passed to a variable creator
      scope.

  Returns:
    A tuple of variable name, shape, dtype, initialization function.
  """
  if (isinstance(kwargs["initial_value"], functools.partial) and (
      "shape" in kwargs["initial_value"].keywords or
      kwargs["initial_value"].args)):
    # Sometimes shape is passed positionally, sometimes it's passed as a kwarg.
    if "shape" in kwargs["initial_value"].keywords:
      shape = kwargs["initial_value"].keywords["shape"]
    else:
      shape = kwargs["initial_value"].args[0]
    return (kwargs["name"], shape,
            kwargs["initial_value"].keywords.get("dtype", kwargs["dtype"]),
            kwargs["initial_value"].func)
  elif "shape" not in kwargs or kwargs["shape"] is None or not callable(
      kwargs["initial_value"]):
    raise ValueError(
        "Unable to extract initializer function and shape from {}. Please "
        "either pass a function that expects a shape and dtype as the "
        "initial value for your variable or functools.partial object with "
        "the shape and dtype kwargs set. This is needed so that we can "
        "initialize the shards of the ShardedVariable locally.".format(
            kwargs["initial_value"]))
  else:
    return (kwargs["name"], kwargs["shape"], kwargs["dtype"],
            kwargs["initial_value"])


def make_sharded_variable_creator(
    hosts: List[Text]) -> Callable[..., TPUEmbeddingVariable]:
  # [FUNCTION DOC]
  # Builds a variable creator that shards a table row-wise across hosts. Input: list of host
  # device names. Returns a creator function suitable for a variable creator scope.
  # The creator splits rows as evenly as possible (earlier hosts get an extra row), creates a
  # variable per non-empty partition on its host via the next creator, with names suffixed by
  # shard index, and supports shard-aware initializers (passing shard info with offsets) or
  # plain ones (called with the shard shape), then wraps all shards in one sharded variable.
  """Makes a sharded variable creator given a list of hosts.

  Args:
    hosts: a list of tensorflow devices on which to shard the tensors.

  Returns:
    A variable creator function.
  """

  def sharded_variable_creator(
      next_creator: Callable[..., tf_variables.Variable], *args, **kwargs):
    # [FUNCTION DOC]
    # The creator returned by the enclosing factory. Inputs: the next creator in the chain
    # plus positional and keyword arguments of a variable creation request. Marks the request
    # to skip the mirrored creator, extracts name, shape, dtype and initializer, partitions
    # the rows over the hosts, creates one shard per non-empty partition on its host
    # (mutating the keyword arguments for each) and returns a sharded TPU embedding variable.
    # Raises a value error (via the extraction helper) when shape or initializer cannot be
    # determined; also assumes a rank two shape.
    """The sharded variable creator."""
    kwargs["skip_mirrored_creator"] = True

    num_hosts = len(hosts)
    name, shape, dtype, unwrapped_initial_value = extract_variable_info(kwargs)
    initial_value = kwargs["initial_value"]
    rows = shape[0]
    cols = shape[1]
    partial_partition = rows % num_hosts
    full_rows_per_host = rows // num_hosts
    # We partition as if we were using MOD sharding: at least
    # `full_rows_per_host` rows to `num_hosts` hosts, where the first
    # `partial_partition` hosts get an additional row when the number of rows
    # is not cleanly divisible. Note that `full_rows_per_host` may be zero.
    partitions = (
        [full_rows_per_host + 1] * partial_partition
        + [full_rows_per_host] * (num_hosts - partial_partition))
    variables = []
    sharding_aware = "shard_info" in tf_inspect.getargspec(initial_value).args

    # Keep track of offset for sharding aware initializers.
    offset = 0
    kwargs["dtype"] = dtype
    for i, p in enumerate(partitions):
      if p == 0:
        # Skip variable creation for empty partitions, resulting from the edge
        # case of 'rows < num_hosts'. This is safe because both load/restore
        # can handle the missing values.
        continue
      with ops.device(hosts[i]):
        kwargs["name"] = "{}_{}".format(name, i)
        kwargs["shape"] = (p, cols)
        if sharding_aware:
          shard_info = base.ShardInfo(kwargs["shape"], (offset, 0))
          kwargs["initial_value"] = functools.partial(
              initial_value, shard_info=shard_info)
          offset += p
        else:
          kwargs["initial_value"] = functools.partial(
              unwrapped_initial_value, kwargs["shape"], dtype=dtype)
        variables.append(next_creator(*args, **kwargs))
    return TPUEmbeddingVariable(variables, name=name)
  return sharded_variable_creator
