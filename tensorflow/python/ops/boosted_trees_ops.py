# [MODULE DOC]
# Python-level support for boosted trees ops. Re-exports many generated boosted-trees ops under
# shorter names for use by other modules (the estimator boosted trees code), and defines helper
# classes: a pruning-mode enumeration, a quantile accumulator resource with checkpoint support,
# and a tree ensemble resource wrapper with checkpoint support. Import-time side effects are
# limited to importing the generated ops; resources and saveable registrations happen only when
# the classes are instantiated in a graph.
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
"""Ops for boosted_trees."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources

# Re-exporting ops used by other modules.
# pylint: disable=unused-import
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_bucketize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split as calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split_v2 as calculate_best_feature_split_v2
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_gains_per_feature as calculate_best_gains_per_feature
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_center_bias as center_bias
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_create_quantile_stream_resource as create_quantile_stream_resource
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_example_debug_outputs as example_debug_outputs
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_quantile_summaries as make_quantile_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_stats_summary as make_stats_summary
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_predict as predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_add_summaries as quantile_add_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_deserialize as quantile_resource_deserialize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_flush as quantile_flush
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_get_bucket_boundaries as get_bucket_boundaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_handle_op as quantile_resource_handle_op
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_calculate_best_feature_split as sparse_calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_training_predict as training_predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble as update_ensemble
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble_v2 as update_ensemble_v2
from tensorflow.python.ops.gen_boosted_trees_ops import is_boosted_trees_quantile_stream_resource_initialized as is_quantile_resource_initialized
# pylint: enable=unused-import

from tensorflow.python.training import saver


class PruningMode:
  # [CLASS DOC]
  # Enumeration-like holder of tree pruning modes: no pruning, pre-pruning and post-pruning,
  # exposed as small integer constants, plus a private mapping from lowercase names (none, pre,
  # post) to those constants. Never instantiated; used through the class-level constants and
  # the string conversion method.
  """Class for working with Pruning modes."""
  NO_PRUNING, PRE_PRUNING, POST_PRUNING = range(0, 3)

  _map = {'none': NO_PRUNING, 'pre': PRE_PRUNING, 'post': POST_PRUNING}

  @classmethod
  def from_str(cls, mode):
    # [FUNCTION DOC]
    # Converts a pruning mode name (one of none, pre, post; case-sensitive) to its integer
    # constant and returns it. Raises a value error listing the valid names if the string is
    # not recognized. No side effects.
    if mode in cls._map:
      return cls._map[mode]
    else:
      raise ValueError(
          'pruning_mode mode must be one of: {}. Found: {}'.format(', '.join(
              sorted(cls._map)), mode))


class QuantileAccumulatorSaveable(saver.BaseSaverBuilder.SaveableObject):
  # [CLASS DOC]
  # HELPER: checkpoint glue for the QuantileAccumulator resource.
  # Saveable object that saves the per-stream bucket boundaries of a quantile stream resource
  # as one save spec per stream and restores them by re-creating the resource and deserializing
  # the boundaries. Created by QuantileAccumulator and registered in the saveable collection.
  """SaveableObject implementation for QuantileAccumulator."""

  def __init__(self, resource_handle, create_op, num_streams, name):
    # [FUNCTION DOC]
    # Builds the save specs. Inputs: resource_handle (handle of the quantile stream resource),
    # create_op (op that initializes the resource, run before restore), num_streams (number of
    # feature streams), name (prefix for checkpoint entry names). Reads the bucket boundaries
    # from the resource (adds graph ops) and creates one save spec per stream named with a
    # boundary suffix and the stream index. Returns nothing; stores the handle and create op.
    self.resource_handle = resource_handle
    self._num_streams = num_streams
    self._create_op = create_op
    bucket_boundaries = get_bucket_boundaries(self.resource_handle,
                                              self._num_streams)
    slice_spec = ''
    specs = []

    def make_save_spec(tensor, suffix):
      # [FUNCTION DOC]
      # HELPER: builds save specs inside the saveable constructor.
      # Wraps a tensor into a save spec using the enclosing empty slice spec and the enclosing
      # name plus the given suffix. Returns the new save spec; no other side effects.
      return saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name + suffix)

    for i in range(self._num_streams):
      specs += [
          make_save_spec(bucket_boundaries[i], '_bucket_boundaries_' + str(i))
      ]
    super(QuantileAccumulatorSaveable, self).__init__(self.resource_handle,
                                                      specs, name)

  def restore(self, restored_tensors, unused_tensor_shapes):
    # [FUNCTION DOC]
    # Restores the bucket boundaries. Input restored_tensors is the list of boundary tensors
    # loaded from a checkpoint; the shapes argument is ignored. Under a control dependency on
    # the resource create op, returns the deserialize op that loads the boundaries into the
    # resource.
    bucket_boundaries = restored_tensors
    with ops.control_dependencies([self._create_op]):
      return quantile_resource_deserialize(
          self.resource_handle, bucket_boundaries=bucket_boundaries)


class QuantileAccumulator():
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring calls this a SaveableObject implementation, but it is a plain
  # class; the saveable behavior lives in QuantileAccumulatorSaveable, and the TF2 serialization
  # hooks here raise not-implemented errors.
  # Wrapper around a graph quantile stream resource: creates the resource handle and its
  # initializer, registers it with the resource registry and adds a saveable object to the
  # graph collection so bucket boundaries are checkpointed. Offers methods to add example
  # summaries, flush and read the bucket boundaries.
  """SaveableObject implementation for QuantileAccumulator.

     The bucket boundaries are serialized and deserialized from checkpointing.
  """

  def __init__(self,
               epsilon,
               num_streams,
               num_quantiles,
               name=None,
               max_elements=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: max_elements is accepted but deleted immediately and never used.
    # Creates the quantile stream resource. Inputs: epsilon (approximation error), num_streams
    # (number of feature streams), num_quantiles (buckets to produce on flush), name (optional
    # name scope), max_elements (ignored). Side effects: builds the handle and init ops under a
    # name scope, registers the resource with the resource registry, and adds a saveable object
    # to the graph's saveable-objects collection. Returns nothing.
    del max_elements  # Unused.

    self._eps = epsilon
    self._num_streams = num_streams
    self._num_quantiles = num_quantiles

    with ops.name_scope(name, 'QuantileAccumulator') as name:
      self._name = name
      self.resource_handle = self._create_resource()
      self._init_op = self._initialize()
      is_initialized_op = self.is_initialized()
    resources.register_resource(self.resource_handle, self._init_op,
                                is_initialized_op)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS,
                          QuantileAccumulatorSaveable(
                              self.resource_handle, self._init_op,
                              self._num_streams, self.resource_handle.name))

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: constructor helper. Creates and returns the quantile stream resource handle op
    # using an empty container and the accumulator name as shared name and op name.
    return quantile_resource_handle_op(
        container='', shared_name=self._name, name=self._name)

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: constructor helper. Returns the op that creates the quantile stream resource with
    # the stored epsilon and number of streams.
    return create_quantile_stream_resource(self.resource_handle, self._eps,
                                           self._num_streams)

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property returning the resource initialization op, lazily building it if it is missing
    # (it is always set by the constructor, so the lazy path is effectively unused).
    if self._init_op is None:
      self._init_op = self._initialize()
    return self._init_op

  def is_initialized(self):
    # [FUNCTION DOC]
    # Returns an op yielding a boolean scalar saying whether the quantile stream resource has
    # been initialized.
    return is_quantile_resource_initialized(self.resource_handle)

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # Placeholder for TF2 checkpoint support. Always raises a not-implemented error.
    raise NotImplementedError('When the need arises, TF2 compatibility can be '
                              'added by implementing this method, along with '
                              '_restore_from_tensors below.')

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # Placeholder for TF2 checkpoint support. Takes restored tensors, ignores them and always
    # raises a not-implemented error.
    raise NotImplementedError('When the need arises, TF2 compatibility can be '
                              'added by implementing this method, along with '
                              '_serialize_to_tensors above.')

  def add_summaries(self, float_columns, example_weights):
    # [FUNCTION DOC]
    # Builds quantile summaries from a list of float feature column tensors and the example
    # weights tensor using the stored epsilon, then returns the op that adds those summaries
    # to the resource streams.
    summaries = make_quantile_summaries(float_columns, example_weights,
                                        self._eps)
    summary_op = quantile_add_summaries(self.resource_handle, summaries)
    return summary_op

  def flush(self):
    # [FUNCTION DOC]
    # Returns the op that finalizes the streams and computes the configured number of
    # quantile buckets. Changes resource state when run.
    return quantile_flush(self.resource_handle, self._num_quantiles)

  def get_bucket_boundaries(self):
    # [FUNCTION DOC]
    # Returns the list of bucket boundary tensors, one per stream, read from the resource.
    return get_bucket_boundaries(self.resource_handle, self._num_streams)


class _TreeEnsembleSavable(saver.BaseSaverBuilder.SaveableObject):
  # [CLASS DOC]
  # HELPER: checkpoint glue for the TreeEnsemble resource.
  # Saveable object that saves a tree ensemble as two entries (stamp token and serialized
  # proto) and restores it by deserializing after the create op. Created by TreeEnsemble.
  """SaveableObject implementation for TreeEnsemble."""

  def __init__(self, resource_handle, create_op, name):
    # [FUNCTION DOC]
    # Builds the save specs. Inputs: resource_handle (ensemble resource handle), create_op (op
    # initializing the resource), name (prefix for checkpoint entries). Adds a serialize op to
    # the graph, creates two specs with stamp and serialized suffixes and an empty slice spec,
    # calls the base constructor and stores the handle and create op. Returns nothing.
    """Creates a _TreeEnsembleSavable object.

    Args:
      resource_handle: handle to the decision tree ensemble variable.
      create_op: the op to initialize the variable.
      name: the name to save the tree ensemble variable under.
    """
    stamp_token, serialized = (
        gen_boosted_trees_ops.boosted_trees_serialize_ensemble(resource_handle))
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree ensemble variable. So we just pass an empty
    # value.
    slice_spec = ''
    specs = [
        saver.BaseSaverBuilder.SaveSpec(stamp_token, slice_spec,
                                        name + '_stamp'),
        saver.BaseSaverBuilder.SaveSpec(serialized, slice_spec,
                                        name + '_serialized'),
    ]
    super(_TreeEnsembleSavable, self).__init__(resource_handle, specs, name)
    self.resource_handle = resource_handle
    self._create_op = create_op

  def restore(self, restored_tensors, unused_restored_shapes):
    # [FUNCTION DOC]
    # Restores the ensemble from two loaded tensors (stamp token first, serialized proto
    # second); the shapes argument is ignored. Returns the deserialize op, run under a control
    # dependency on the create op.
    """Restores the associated tree ensemble from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint.
      unused_restored_shapes: the shapes this object should conform to after
        restore. Not meaningful for trees.

    Returns:
      The operation that restores the state of the tree ensemble variable.
    """
    with ops.control_dependencies([self._create_op]):
      return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(
          self.resource_handle,
          stamp_token=restored_tensors[0],
          tree_ensemble_serialized=restored_tensors[1])


class TreeEnsemble():
  # [CLASS DOC]
  # Wrapper around a boosted trees ensemble resource. Creates the resource handle and its
  # create op, registers the resource (shared unless local), and for non-local ensembles adds
  # a saveable object so the ensemble is checkpointed. Provides accessors for stamp token and
  # states, and serialize/deserialize methods.
  """Creates TreeEnsemble resource."""

  def __init__(self, name, stamp_token=0, is_local=False, serialized_proto=''):
    # [FUNCTION DOC]
    # Creates the ensemble resource. Inputs: name (name scope), stamp_token (initial stamp,
    # default zero), is_local (if true the resource is not shared or checkpointed, default
    # false), serialized_proto (initial serialized ensemble, default empty string). Side
    # effects: builds handle and create ops, registers the resource with the resource
    # registry, and for non-local ensembles adds a saveable object to the graph collection.
    self._stamp_token = stamp_token
    self._serialized_proto = serialized_proto
    self._is_local = is_local
    with ops.name_scope(name, 'TreeEnsemble') as name:
      self._name = name
      self.resource_handle = self._create_resource()
      self._init_op = self._initialize()
      is_initialized_op = self.is_initialized()
      # Adds the variable to the savable list.
      if not is_local:
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS,
                              _TreeEnsembleSavable(
                                  self.resource_handle, self.initializer,
                                  self.resource_handle.name))
      resources.register_resource(
          self.resource_handle,
          self.initializer,
          is_initialized_op,
          is_shared=not is_local)

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: constructor helper. Returns the ensemble resource handle op with an empty
    # container and the ensemble name as shared name and op name.
    return gen_boosted_trees_ops.boosted_trees_ensemble_resource_handle_op(
        container='', shared_name=self._name, name=self._name)

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: constructor helper. Returns the op creating the ensemble with the stored stamp
    # token and serialized proto.
    return gen_boosted_trees_ops.boosted_trees_create_ensemble(
        self.resource_handle,
        self._stamp_token,
        tree_ensemble_serialized=self._serialized_proto)

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property returning the ensemble create op, lazily building it if missing (the
    # constructor always sets it, so the lazy path is effectively unused).
    if self._init_op is None:
      self._init_op = self._initialize()
    return self._init_op

  def is_initialized(self):
    # [FUNCTION DOC]
    # Returns an op yielding a boolean scalar saying whether the ensemble resource has been
    # initialized.
    return gen_boosted_trees_ops.is_boosted_trees_ensemble_initialized(
        self.resource_handle)

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # Placeholder for TF2 checkpoint support. Always raises a not-implemented error.
    raise NotImplementedError('When the need arises, TF2 compatibility can be '
                              'added by implementing this method, along with '
                              '_restore_from_tensors below.')

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # Placeholder for TF2 checkpoint support. Takes restored tensors, ignores them and always
    # raises a not-implemented error.
    raise NotImplementedError('When the need arises, TF2 compatibility can be '
                              'added by implementing this method, along with '
                              '_serialize_to_tensors above.')

  def get_stamp_token(self):
    # [FUNCTION DOC]
    # Reads the ensemble states from the resource and returns only the stamp token tensor.
    """Returns the current stamp token of the resource."""
    stamp_token, _, _, _, _ = (
        gen_boosted_trees_ops.boosted_trees_get_ensemble_states(
            self.resource_handle))
    return stamp_token

  def get_states(self):
    # [FUNCTION DOC]
    # Reads the ensemble states and returns a tuple of five named identity tensors: stamp
    # token, number of trees, number of finalized trees, number of attempted layers and the
    # node range of the latest layer.
    """Returns states of the tree ensemble.

    Returns:
      stamp_token, num_trees, num_finalized_trees, num_attempted_layers and
      range of the nodes in the latest layer.
    """
    (stamp_token, num_trees, num_finalized_trees, num_attempted_layers,
     nodes_range) = (
         gen_boosted_trees_ops.boosted_trees_get_ensemble_states(
             self.resource_handle))
    # Use identity to give names.
    return (array_ops.identity(stamp_token, name='stamp_token'),
            array_ops.identity(num_trees, name='num_trees'),
            array_ops.identity(num_finalized_trees, name='num_finalized_trees'),
            array_ops.identity(
                num_attempted_layers, name='num_attempted_layers'),
            array_ops.identity(nodes_range, name='last_layer_nodes_range'))

  def serialize(self):
    # [FUNCTION DOC]
    # Returns the pair of tensors (stamp token, serialized ensemble proto string) produced by
    # the serialize op for this resource.
    """Serializes the ensemble into proto and returns the serialized proto.

    Returns:
      stamp_token: int64 scalar Tensor to denote the stamp of the resource.
      serialized_proto: string scalar Tensor of the serialized proto.
    """
    return gen_boosted_trees_ops.boosted_trees_serialize_ensemble(
        self.resource_handle)

  def deserialize(self, stamp_token, serialized_proto):
    # [FUNCTION DOC]
    # Returns the op that resets the ensemble resource from the given stamp token (int64
    # scalar tensor) and serialized proto (string scalar tensor); running it mutates the
    # resource.
    """Deserialize the input proto and resets the ensemble from it.

    Args:
      stamp_token: int64 scalar Tensor to denote the stamp of the resource.
      serialized_proto: string scalar Tensor of the serialized proto.

    Returns:
      Operation (for dependencies).
    """
    return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(
        self.resource_handle, stamp_token, serialized_proto)
