# [MODULE DOC]
# Graph-building k-means clustering operations. Defines the KMeans class, which builds training
# and inference graph pieces (full-batch Lloyd iterations or mini-batch updates, with squared
# Euclidean or cosine distance), plus an internal factory that builds the op choosing initial
# cluster centers (random, kmeans plus plus, k-MC2, a user function, or explicit values). Also
# defines string constants naming distance metrics, initialization methods and the cluster
# variable name, and re-exports every generated clustering op via a wildcard import. Import-time
# side effects: none beyond importing the generated op module.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Clustering Operations."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as random_seed_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_clustering_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.embedding_ops import embedding_lookup
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_clustering_ops import *
# pylint: enable=wildcard-import

# Euclidean distance between vectors U and V is defined as \\(||U - V||_F\\)
# which is the square root of the sum of the absolute squares of the elements
# difference.
SQUARED_EUCLIDEAN_DISTANCE = 'squared_euclidean'
# Cosine distance between vectors U and V is defined as
# \\(1 - (U \dot V) / (||U||_F ||V||_F)\\)
COSINE_DISTANCE = 'cosine'

RANDOM_INIT = 'random'
KMEANS_PLUS_PLUS_INIT = 'kmeans_plus_plus'
KMC2_INIT = 'kmc2'

CLUSTERS_VAR_NAME = 'clusters'


class KMeans:
  # [CLASS DOC]
  # Builds the TensorFlow graph for k-means clustering. Holds the (list of) input tensors, number
  # of clusters, initialization choice, distance metric, mini-batch settings, a derived random
  # seed and the settings for the kmeans plus plus and k-MC2 initializers. The main entry point is
  # the training graph method, which creates variables for the centers and returns scores, cluster
  # indices, an initialization op and a training op. Supports full-batch and mini-batch variants.
  # Private helpers compute distances and build update ops.
  """Creates the graph for k-means clustering."""

  def __init__(self,
               inputs,
               num_clusters,
               initial_clusters=RANDOM_INIT,
               distance_metric=SQUARED_EUCLIDEAN_DISTANCE,
               use_mini_batch=False,
               mini_batch_steps_per_iteration=1,
               random_seed=0,
               kmeans_plus_plus_num_retries=2,
               kmc2_chain_length=200):
    # [FUNCTION DOC]
    # Validates and stores the clustering configuration; builds no graph ops. inputs is a tensor
    # or list of tensors (a single tensor is wrapped in a list). num_clusters is the cluster count.
    # initial_clusters is a method name string, a function, or initial center values. Only string
    # values are validated against the allowed method names. distance_metric must be squared
    # Euclidean or cosine. use_mini_batch selects the mini-batch variant;
    # mini_batch_steps_per_iteration is converted to an integer. random_seed is turned into an
    # operation-level seed through the framework seed helper (reads graph-level seed state). The
    # kmeans plus plus retry count and k-MC2 chain length are stored. Raises a value error for an
    # unsupported initialization name or distance metric.
    """Creates an object for generating KMeans clustering graph.

    This class implements the following variants of K-means algorithm:

    If use_mini_batch is False, it runs standard full batch K-means. Each step
    runs a single iteration of K-Means. This step can be run sharded across
    multiple workers by passing a list of sharded inputs to this class. Note
    however that a single step needs to process the full input at once.

    If use_mini_batch is True, it runs a generalization of the mini-batch
    K-means algorithm. It runs multiple iterations, where each iteration is
    composed of mini_batch_steps_per_iteration steps. Two copies of cluster
    centers are maintained: one that is updated at the end of each iteration,
    and one that is updated every step. The first copy is used to compute
    cluster allocations for each step, and for inference, while the second copy
    is the one updated each step using the mini-batch update rule. After each
    iteration is complete, this second copy is copied back the first copy.

    Note that for use_mini_batch=True, when mini_batch_steps_per_iteration=1,
    the algorithm reduces to the standard mini-batch algorithm. Also by setting
    mini_batch_steps_per_iteration = num_inputs / batch_size, the algorithm
    becomes an asynchronous version of the full-batch algorithm. Note however
    that there is no guarantee by this implementation that each input is seen
    exactly once per iteration. Also, different updates are applied
    asynchronously without locking. So this asynchronous version may not behave
    exactly like a full-batch version.

    Args:
      inputs: An input tensor or list of input tensors. It is assumed that the
        data points have been previously randomly permuted.
      num_clusters: An integer tensor specifying the number of clusters. This
        argument is ignored if initial_clusters is a tensor or numpy array.
      initial_clusters: Specifies the clusters used during initialization. One
        of the following: - a tensor or numpy array with the initial cluster
          centers. - a function f(inputs, k) that returns up to k centers from
          `inputs`.
        - "random": Choose centers randomly from `inputs`.
        - "kmeans_plus_plus": Use kmeans++ to choose centers from `inputs`.
        - "kmc2": Use the fast k-MC2 algorithm to choose centers from `inputs`.
          In the last three cases, one batch of `inputs` may not yield
          `num_clusters` centers, in which case initialization will require
          multiple batches until enough centers are chosen. In the case of
          "random" or "kmeans_plus_plus", if the input size is <= `num_clusters`
          then the entire batch is chosen to be cluster centers.
      distance_metric: Distance metric used for clustering. Supported options:
        "squared_euclidean", "cosine".
      use_mini_batch: If true, use the mini-batch k-means algorithm. Else assume
        full batch.
      mini_batch_steps_per_iteration: Number of steps after which the updated
        cluster centers are synced back to a master copy.
      random_seed: Seed for PRNG used to initialize seeds.
      kmeans_plus_plus_num_retries: For each point that is sampled during
        kmeans++ initialization, this parameter specifies the number of
        additional points to draw from the current distribution before selecting
        the best. If a negative value is specified, a heuristic is used to
        sample O(log(num_to_sample)) additional points.
      kmc2_chain_length: Determines how many candidate points are used by the
        k-MC2 algorithm to produce one new cluster centers. If a (mini-)batch
        contains less points, one new cluster center is generated from the
        (mini-)batch.

    Raises:
      ValueError: An invalid argument was passed to initial_clusters or
        distance_metric.
    """
    initialization_algorithms = [RANDOM_INIT, KMEANS_PLUS_PLUS_INIT, KMC2_INIT]
    if isinstance(initial_clusters,
                  str) and initial_clusters not in initialization_algorithms:
      raise ValueError(
          f'Unsupported initialization algorithm `{initial_clusters}`,'
          f'must be one of `{initialization_algorithms}`.')

    distance_metrics = [SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE]
    if distance_metric not in distance_metrics:
      raise ValueError(f'Unsupported distance metric `{distance_metric}`,'
                       f'must be one of `{distance_metrics}`.')
    self._inputs = inputs if isinstance(inputs, list) else [inputs]
    self._num_clusters = num_clusters
    self._initial_clusters = initial_clusters
    self._distance_metric = distance_metric
    self._use_mini_batch = use_mini_batch
    self._mini_batch_steps_per_iteration = int(mini_batch_steps_per_iteration)
    self._seed = random_seed_ops.get_seed(random_seed)[0]
    self._kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries
    self._kmc2_chain_length = kmc2_chain_length

  @classmethod
  def _distance_graph(cls, inputs, clusters, distance_metric):
    # [FUNCTION DOC]
    # HELPER: dispatcher that picks the distance computation for KMeans graphs.
    # Class method. inputs must be a list of tensors (asserted); clusters is the center matrix.
    # Returns, for squared Euclidean, the list from the Euclidean helper; for cosine, the list from
    # the cosine helper assuming normalized inputs. Any other metric trips an assertion failure
    # rather than a value error.
    """Computes distance between each input and each cluster center.

    Args:
      inputs: list of input Tensors.
      clusters: cluster Tensor.
      distance_metric: distance metric used for clustering

    Returns:
      list of Tensors, where each element corresponds to each element in inputs.
      The value is the distance of each row to all the cluster centers.
      Currently only Euclidean distance and cosine distance are supported.
    """
    assert isinstance(inputs, list)
    if distance_metric == SQUARED_EUCLIDEAN_DISTANCE:
      return cls._compute_euclidean_distance(inputs, clusters)
    elif distance_metric == COSINE_DISTANCE:
      return cls._compute_cosine_distance(
          inputs, clusters, inputs_normalized=True)
    else:
      assert False, str(distance_metric)

  @classmethod
  def _compute_euclidean_distance(cls, inputs, clusters):
    # [FUNCTION DOC]
    # HELPER: squared Euclidean distance computation for KMeans.
    # KNOWN ISSUE: the docstring says Euclidean distance, but the values returned are squared
    # distances (no square root is taken).
    # For each input matrix, computes the matrix of squared distances from every row to every
    # cluster center using the expansion of squared norms minus twice the dot product, on the
    # device of that input. Returns a list with one distance matrix per input. Class method.
    """Computes Euclidean distance between each input and each cluster center.

    Args:
      inputs: list of input Tensors.
      clusters: cluster Tensor.

    Returns:
      list of Tensors, where each element corresponds to each element in inputs.
      The value is the distance of each row to all the cluster centers.
    """
    output = []
    for inp in inputs:
      with ops.colocate_with(inp, ignore_existing=True):
        # Computes Euclidean distance. Note the first and third terms are
        # broadcast additions.
        squared_distance = (
            math_ops.reduce_sum(math_ops.square(inp), 1, keepdims=True) -
            2 * math_ops.matmul(inp, clusters, transpose_b=True) +
            array_ops.transpose(
                math_ops.reduce_sum(
                    math_ops.square(clusters), 1, keepdims=True)))
        output.append(squared_distance)

    return output

  @classmethod
  def _compute_cosine_distance(cls, inputs, clusters, inputs_normalized=True):
    # [FUNCTION DOC]
    # HELPER: cosine distance computation for KMeans.
    # Class method. For each input matrix computes one minus the dot product with the cluster
    # centers. If inputs_normalized is false, the clusters and each input are first L2-normalized
    # along rows; otherwise they are assumed already normalized. Returns a list with one distance
    # matrix per input. Computations are placed on the device of the respective tensor.
    """Computes cosine distance between each input and each cluster center.

    Args:
      inputs: list of input Tensor.
      clusters: cluster Tensor
      inputs_normalized: if True, it assumes that inp and clusters are
        normalized and computes the dot product which is equivalent to the
        cosine distance. Else it L2 normalizes the inputs first.

    Returns:
      list of Tensors, where each element corresponds to each element in inp.
      The value is the distance of each row to all the cluster centers.
    """
    output = []
    if not inputs_normalized:
      with ops.colocate_with(clusters, ignore_existing=True):
        clusters = nn_impl.l2_normalize(clusters, axis=1)
    for inp in inputs:
      with ops.colocate_with(inp, ignore_existing=True):
        if not inputs_normalized:
          inp = nn_impl.l2_normalize(inp, axis=1)
        output.append(1 - math_ops.matmul(inp, clusters, transpose_b=True))
    return output

  def _infer_graph(self, inputs, clusters):
    # [FUNCTION DOC]
    # HELPER: builds the nearest-center inference sub-graph for the training graph.
    # KNOWN ISSUE: the docstring says a list of tuples is returned, but the code returns a lazy
    # zip transposition, i.e. three sequences (all scores, closest distances, closest indices).
    # inputs is a list of tensors (asserted); clusters is the center matrix. Computes pairwise
    # distances, and for cosine distance with centers not kept normalized, normalizes the centers.
    # Then, per input, finds the nearest center using the nearest-neighbors op, halves distances
    # for cosine, and squeezes the last dimension. Returns the transposed per-input results.
    """Maps input to closest cluster and the score.

    Args:
      inputs: list of input Tensors.
      clusters: Tensor of cluster centers.

    Returns:
      List of tuple, where each value in tuple corresponds to a value in inp.
      The tuple has following three elements:
      all_scores: distance of each input to each cluster center.
      score: distance of each input to closest cluster center.
      cluster_idx: index of cluster center closest to the corresponding input.
    """
    assert isinstance(inputs, list)
    # Pairwise distances are used only by transform(). In all other cases, this
    # sub-graph is not evaluated.
    scores = self._distance_graph(inputs, clusters, self._distance_metric)
    output = []
    if (self._distance_metric == COSINE_DISTANCE and
        not self._clusters_l2_normalized()):
      # The cosine distance between normalized vectors x and y is the same as
      # 2 * squared_euclidean_distance. We are using this fact and reusing the
      # nearest_neighbors op.
      # TODO(ands): Support COSINE distance in nearest_neighbors and remove
      # this.
      with ops.colocate_with(clusters, ignore_existing=True):
        clusters = nn_impl.l2_normalize(clusters, axis=1)
    for inp, score in zip(inputs, scores):
      with ops.colocate_with(inp, ignore_existing=True):
        (indices,
         distances) = gen_clustering_ops.nearest_neighbors(inp, clusters, 1)
        if self._distance_metric == COSINE_DISTANCE:
          distances *= 0.5
        output.append(
            (score, array_ops.squeeze(distances,
                                      [-1]), array_ops.squeeze(indices, [-1])))
    return zip(*output)

  def _clusters_l2_normalized(self):
    # [FUNCTION DOC]
    # HELPER: predicate telling whether stored cluster centers are kept L2-normalized.
    # Returns true only for the cosine metric when not in mini-batch mode, or when mini-batch
    # mode syncs more than one step per iteration; false otherwise. No side effects.
    """Returns True if clusters centers are kept normalized."""
    return (self._distance_metric == COSINE_DISTANCE and
            (not self._use_mini_batch or
             self._mini_batch_steps_per_iteration > 1))

  def _create_variables(self, num_clusters):
    # [FUNCTION DOC]
    # HELPER: creates the state variables used by the training graph.
    # num_clusters is an integer tensor. Creates a shape-unvalidated variable for the cluster
    # centers (initially empty) and a boolean initialized flag. In mini-batch mode with more than
    # one step per iteration, also creates an updated copy of the centers, a step countdown and
    # zero-initialized counts. Otherwise the updated copy aliases the centers, the countdown is
    # nothing, and counts are ones in mini-batch mode or nothing otherwise. Returns a five-tuple:
    # centers, initialized flag, counts, updated centers, steps-left. Side effect: adds variables.
    """Creates variables.

    Args:
      num_clusters: an integer Tensor providing the number of clusters.

    Returns:
      Tuple with following elements:
      - cluster_centers: a Tensor for storing cluster centers
      - cluster_centers_initialized: bool Variable indicating whether clusters
            are initialized.
      - cluster_counts: a Tensor for storing counts of points assigned to this
            cluster. This is used by mini-batch training.
      - cluster_centers_updated: Tensor representing copy of cluster centers
            that are updated every step.
      - update_in_steps: numbers of steps left before we sync
            cluster_centers_updated back to cluster_centers.
    """
    init_value = array_ops.placeholder_with_default([], shape=None)
    cluster_centers = variable_v1.VariableV1(
        init_value, name=CLUSTERS_VAR_NAME, validate_shape=False)
    cluster_centers_initialized = variable_v1.VariableV1(
        False, dtype=dtypes.bool, name='initialized')

    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
      # Copy of cluster centers actively updated each step according to
      # mini-batch update rule.
      cluster_centers_updated = variable_v1.VariableV1(
          init_value, name='clusters_updated', validate_shape=False)
      # How many steps till we copy the updated clusters to cluster_centers.
      update_in_steps = variable_v1.VariableV1(
          self._mini_batch_steps_per_iteration,
          dtype=dtypes.int64,
          name='update_in_steps')
      # Count of points assigned to cluster_centers_updated.
      cluster_counts = variable_v1.VariableV1(
          array_ops.zeros([num_clusters], dtype=dtypes.int64))
    else:
      cluster_centers_updated = cluster_centers
      update_in_steps = None
      cluster_counts = (
          variable_v1.VariableV1(
              array_ops.ones([num_clusters], dtype=dtypes.int64))
          if self._use_mini_batch else None)
    return (cluster_centers, cluster_centers_initialized, cluster_counts,
            cluster_centers_updated, update_in_steps)

  @classmethod
  def _l2_normalize_data(cls, inputs):
    # [FUNCTION DOC]
    # HELPER: row-wise L2 normalization of every input tensor in a list.
    # Class method; returns a new list of normalized tensors, each computed on its input's device.
    """Normalized the input data."""
    output = []
    for inp in inputs:
      with ops.colocate_with(inp, ignore_existing=True):
        output.append(nn_impl.l2_normalize(inp, dim=1))
    return output

  def training_graph(self):
    # [FUNCTION DOC]
    # Builds the full k-means graph. No arguments. Resolves the initial clusters (strings and
    # functions keep the configured cluster count as a tensor; explicit values are converted to a
    # tensor and define the count). Creates the variables, builds the initialization op through
    # the internal factory, normalizes the data for cosine distance, builds the inference graph,
    # and then builds either the mini-batch training op (gated by the sync-updates op through a
    # control dependency) or the full-batch training op. Returns a six-tuple: all scores, cluster
    # indices, closest scores, initialized flag variable, init op, training op. Side effects:
    # creates variables and ops in the current graph.
    """Generate a training graph for kmeans algorithm.

    This returns, among other things, an op that chooses initial centers
    (init_op), a boolean variable that is set to True when the initial centers
    are chosen (cluster_centers_initialized), and an op to perform either an
    entire Lloyd iteration or a mini-batch of a Lloyd iteration (training_op).
    The caller should use these components as follows. A single worker should
    execute init_op multiple times until cluster_centers_initialized becomes
    True. Then multiple workers may execute training_op any number of times.

    Returns:
      A tuple consisting of:
      all_scores: A matrix (or list of matrices) of dimensions (num_input,
        num_clusters) where the value is the distance of an input vector and a
        cluster center.
      cluster_idx: A vector (or list of vectors). Each element in the vector
        corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      scores: Similar to cluster_idx but specifies the distance to the
        assigned cluster instead.
      cluster_centers_initialized: scalar indicating whether clusters have been
        initialized.
      init_op: an op to initialize the clusters.
      training_op: an op that runs an iteration of training.
    """
    # Implementation of kmeans.
    if (isinstance(self._initial_clusters, str) or
        callable(self._initial_clusters)):
      initial_clusters = self._initial_clusters
      num_clusters = ops.convert_to_tensor(self._num_clusters)
    else:
      initial_clusters = ops.convert_to_tensor(self._initial_clusters)
      num_clusters = array_ops.shape(initial_clusters)[0]

    inputs = self._inputs
    (cluster_centers_var, cluster_centers_initialized, total_counts,
     cluster_centers_updated,
     update_in_steps) = self._create_variables(num_clusters)
    init_op = _InitializeClustersOpFactory(
        self._inputs, num_clusters, initial_clusters, self._distance_metric,
        self._seed, self._kmeans_plus_plus_num_retries, self._kmc2_chain_length,
        cluster_centers_var, cluster_centers_updated,
        cluster_centers_initialized).op()
    cluster_centers = cluster_centers_var

    if self._distance_metric == COSINE_DISTANCE:
      inputs = self._l2_normalize_data(inputs)
      if not self._clusters_l2_normalized():
        cluster_centers = nn_impl.l2_normalize(cluster_centers, dim=1)

    all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
    if self._use_mini_batch:
      sync_updates_op = self._mini_batch_sync_updates_op(
          update_in_steps, cluster_centers_var, cluster_centers_updated,
          total_counts)
      assert sync_updates_op is not None
      with ops.control_dependencies([sync_updates_op]):
        training_op = self._mini_batch_training_op(inputs, cluster_idx,
                                                   cluster_centers_updated,
                                                   total_counts)
    else:
      assert cluster_centers == cluster_centers_var
      training_op = self._full_batch_training_op(inputs, num_clusters,
                                                 cluster_idx,
                                                 cluster_centers_var)

    return (all_scores, cluster_idx, scores, cluster_centers_initialized,
            init_op, training_op)

  def _mini_batch_sync_updates_op(self, update_in_steps, cluster_centers_var,
                                  cluster_centers_updated, total_counts):
    # [FUNCTION DOC]
    # HELPER: builds the op that periodically syncs the per-step updated centers back to the main
    # center variable in mini-batch mode.
    # If mini-batch mode is on with more than one step per iteration, returns a conditional op: when
    # the countdown reaches zero, it resets the countdown, copies (normalizing for cosine) the
    # updated centers into the center variable and zeroes the counts; otherwise it decrements the
    # countdown. In every other configuration returns a no-op. Inherently racy across workers
    # (best effort).
    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
      assert update_in_steps is not None
      with ops.colocate_with(update_in_steps, ignore_existing=True):

        def _f():
          # [FUNCTION DOC]
          # HELPER: branch of the sync condition taken when the countdown has expired.
          # Resets the countdown to the steps-per-iteration minus one, then copies the updated
          # centers (L2-normalized for cosine) into the center variable, then zeroes the
          # counts, ordered by control dependencies. Returns the countdown value after those
          # side effects. Mutates the countdown, centers and counts variables.
          # Note that there is a race condition here, so we do a best effort
          # updates here. We reset update_in_steps first so that other workers
          # don't duplicate the updates. Also we update cluster_center_vars
          # before resetting total_counts to avoid large updates to
          # cluster_centers_updated based on partially updated
          # cluster_center_vars.
          with ops.control_dependencies([
              state_ops.assign(update_in_steps,
                               self._mini_batch_steps_per_iteration - 1)
          ]):
            with ops.colocate_with(
                cluster_centers_updated, ignore_existing=True):
              if self._distance_metric == COSINE_DISTANCE:
                cluster_centers = nn_impl.l2_normalize(
                    cluster_centers_updated, dim=1)
              else:
                cluster_centers = cluster_centers_updated
            with ops.colocate_with(cluster_centers_var, ignore_existing=True):
              with ops.control_dependencies(
                  [state_ops.assign(cluster_centers_var, cluster_centers)]):
                with ops.colocate_with(None, ignore_existing=True):
                  with ops.control_dependencies([
                      state_ops.assign(total_counts,
                                       array_ops.zeros_like(total_counts))
                  ]):
                    return array_ops.identity(update_in_steps)

        return cond.cond(
            update_in_steps <= 0, _f,
            lambda: state_ops.assign_sub(update_in_steps, 1))
    else:
      return control_flow_ops.no_op()

  def _mini_batch_training_op(self, inputs, cluster_idx_list, cluster_centers,
                              total_counts):
    # [FUNCTION DOC]
    # HELPER: builds the mini-batch k-means update op.
    # For each input and its cluster assignment vector: deduplicates assigned cluster ids, gathers
    # old counts and centers, aggregates new point counts and point sums per cluster, and computes
    # the per-cluster increment as the sum minus count times old center, scaled by one over
    # (old count plus new count). Then scatter-adds the counts and the center increments into the
    # variables. total_counts must not be nothing (asserted). Returns a grouped op running all
    # updates. Mutates the counts and centers variables when run.
    """Creates an op for training for mini batch case.

    Args:
      inputs: list of input Tensors.
      cluster_idx_list: A vector (or list of vectors). Each element in the
        vector corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      cluster_centers: Tensor Ref of cluster centers.
      total_counts: Tensor Ref of cluster counts.

    Returns:
      An op for doing an update of mini-batch k-means.
    """
    update_ops = []
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
      with ops.colocate_with(inp, ignore_existing=True):
        assert total_counts is not None
        cluster_idx = array_ops.reshape(cluster_idx, [-1])
        # Dedupe the unique ids of cluster_centers being updated so that updates
        # can be locally aggregated.
        unique_ids, unique_idx = array_ops.unique(cluster_idx)
        num_unique_cluster_idx = array_ops.size(unique_ids)
        # Fetch the old values of counts and cluster_centers.
        with ops.colocate_with(total_counts, ignore_existing=True):
          old_counts = array_ops.gather(total_counts, unique_ids)
        # TODO(agarwal): This colocation seems to run into problems. Fix it.
        with ops.colocate_with(cluster_centers, ignore_existing=True):
          old_cluster_centers = array_ops.gather(cluster_centers, unique_ids)
        # Locally aggregate the increment to counts.
        count_updates = math_ops.unsorted_segment_sum(
            array_ops.ones_like(unique_idx, dtype=total_counts.dtype),
            unique_idx, num_unique_cluster_idx)
        # Locally compute the sum of inputs mapped to each id.
        # For a cluster with old cluster value x, old count n, and with data
        # d_1,...d_k newly assigned to it, we recompute the new value as
        # \\(x += (sum_i(d_i) - k * x) / (n + k)\\).
        # Compute \\(sum_i(d_i)\\), see comment above.
        cluster_center_updates = math_ops.unsorted_segment_sum(
            inp, unique_idx, num_unique_cluster_idx)
        # Shape to enable broadcasting count_updates and learning_rate to inp.
        # It extends the shape with 1's to match the rank of inp.
        broadcast_shape = array_ops.concat([
            array_ops.reshape(num_unique_cluster_idx, [1]),
            array_ops.ones(
                array_ops.reshape(array_ops.rank(inp) - 1, [1]),
                dtype=dtypes.int32)
        ], 0)
        # Subtract k * x, see comment above.
        cluster_center_updates -= math_ops.cast(
            array_ops.reshape(count_updates, broadcast_shape),
            inp.dtype) * old_cluster_centers
        learning_rate = math_ops.reciprocal(
            math_ops.cast(old_counts + count_updates, inp.dtype))
        learning_rate = array_ops.reshape(learning_rate, broadcast_shape)
        # scale by 1 / (n + k), see comment above.
        cluster_center_updates *= learning_rate
        # Apply the updates.
      update_counts = state_ops.scatter_add(total_counts, unique_ids,
                                            count_updates)
      update_cluster_centers = state_ops.scatter_add(cluster_centers,
                                                     unique_ids,
                                                     cluster_center_updates)
      update_ops.extend([update_counts, update_cluster_centers])
    return control_flow_ops.group(*update_ops)

  def _full_batch_training_op(self, inputs, num_clusters, cluster_idx_list,
                              cluster_centers):
    # [FUNCTION DOC]
    # HELPER: builds the full-batch Lloyd iteration update op.
    # KNOWN ISSUE: the docstring says it returns an op for mini-batch k-means, but this builds the
    # full-batch update (cluster means replacing the centers).
    # Sums the points and counts the points assigned to each cluster across all input shards,
    # divides sums by counts plus a small epsilon, L2-normalizes the new centers if centers are
    # kept normalized, and returns an assignment op writing them into the center variable.
    """Creates an op for training for full batch case.

    Args:
      inputs: list of input Tensors.
      num_clusters: an integer Tensor providing the number of clusters.
      cluster_idx_list: A vector (or list of vectors). Each element in the
        vector corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      cluster_centers: Tensor Ref of cluster centers.

    Returns:
      An op for doing an update of mini-batch k-means.
    """
    cluster_sums = []
    cluster_counts = []
    epsilon = constant_op.constant(1e-6, dtype=inputs[0].dtype)
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
      with ops.colocate_with(inp, ignore_existing=True):
        cluster_sums.append(
            math_ops.unsorted_segment_sum(inp, cluster_idx, num_clusters))
        cluster_counts.append(
            math_ops.unsorted_segment_sum(
                array_ops.reshape(
                    array_ops.ones(
                        array_ops.reshape(array_ops.shape(inp)[0], [-1])),
                    [-1, 1]), cluster_idx, num_clusters))
    with ops.colocate_with(cluster_centers, ignore_existing=True):
      new_clusters_centers = math_ops.add_n(cluster_sums) / (
          math_ops.cast(math_ops.add_n(cluster_counts), cluster_sums[0].dtype) +
          epsilon)
      if self._clusters_l2_normalized():
        new_clusters_centers = nn_impl.l2_normalize(new_clusters_centers, dim=1)
    return state_ops.assign(cluster_centers, new_clusters_centers)


class _InitializeClustersOpFactory:
  # [CLASS DOC]
  # HELPER: internal factory used by the k-means training graph.
  # Builds the single op that adds up to the remaining number of initial cluster centers to the
  # center variable and sets the initialized flag once enough centers exist. Holds the inputs,
  # target cluster count, initialization choice, seed, retry settings, and the center, updated
  # center and initialized variables, plus derived tensors for selected, remaining and data counts.
  # Supports random, kmeans plus plus, k-MC2, user function and explicit center initialization.
  """Internal class to create the op to initialize the clusters.

    The op performs this algorithm (see constructor args):

    num_remaining = num_clusters - length(cluster_centers)
    if num_remaining == 0:
      assert that cluster_centers_initialized is true
    else:
      assert that num_remaining > 0
      new_centers = choose up to num_remaining initial centers
      l2-normalize new_centers if using cosine distance
      all_centers = concat(cluster_centers, new_centers)
      cluster_centers := all_centers
      if there is a cluster_centers_updated variable:
        cluster_centers_updated := cluster_centers
      num_now_remaining = num_clusters - length(cluster_centers)
      if num_now_remaining == 0:
        cluster_centers_initialized := true
  """

  # TODO(ccolby): Refactor this class so that kmc2 isn't so much a special case.

  def __init__(self, inputs, num_clusters, initial_clusters, distance_metric,
               random_seed, kmeans_plus_plus_num_retries, kmc2_chain_length,
               cluster_centers, cluster_centers_updated,
               cluster_centers_initialized):
    # [FUNCTION DOC]
    # Stores all configuration (see the KMeans constructor for the shared parameters) plus the
    # center, updated-center and initialized variables, and derives tensors: the number of centers
    # already selected (rows of the center variable), the number remaining, and the total number
    # of data points across input shards. Adds shape and arithmetic ops to the graph; raises
    # nothing itself.
    """Creates an op factory.

    Args:
      inputs: See KMeans constructor.
      num_clusters: An integer Tensor providing the number of clusters.
      initial_clusters: See KMeans constructor.
      distance_metric: See KMeans constructor.
      random_seed: See KMeans constructor.
      kmeans_plus_plus_num_retries: See KMeans constructor.
      kmc2_chain_length: See KMeans constructor.
      cluster_centers: The TF variable holding the initial centers. It may
        already contain some centers when the op is executed.
      cluster_centers_updated: A second TF variable to hold a copy of the
        initial centers, used for full-batch mode. In mini-batch mode,
        cluster_centers_updated is the same variable as cluster_centers.
      cluster_centers_initialized: A boolean TF variable that will be set to
        true when all the initial centers have been chosen.
    """
    # All of these instance variables are constants.
    self._inputs = inputs
    self._num_clusters = num_clusters
    self._initial_clusters = initial_clusters
    self._distance_metric = distance_metric
    self._seed = random_seed
    self._kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries
    self._kmc2_chain_length = kmc2_chain_length
    self._cluster_centers = cluster_centers
    self._cluster_centers_updated = cluster_centers_updated
    self._cluster_centers_initialized = cluster_centers_initialized

    self._num_selected = array_ops.shape(self._cluster_centers)[0]
    self._num_remaining = self._num_clusters - self._num_selected
    self._num_data = math_ops.add_n(
        [array_ops.shape(i)[0] for i in self._inputs])

  def _random(self):
    # [FUNCTION DOC]
    # HELPER: random sampling initializer.
    # Draws as many uniform random indices (with the stored seed) as centers remain, over the total
    # data size, and looks up those rows across the input shards using the division partition
    # strategy. Returns the sampled rows. Sampling is with replacement.
    indices = random_ops.random_uniform(
        array_ops.reshape(self._num_remaining, [-1]),
        minval=0,
        maxval=math_ops.cast(self._num_data, dtypes.int64),
        seed=self._seed,
        dtype=dtypes.int64)
    return embedding_lookup(self._inputs, indices, partition_strategy='div')

  def _kmeans_plus_plus(self):
    # [FUNCTION DOC]
    # HELPER: kmeans plus plus initializer.
    # KNOWN ISSUE: only the first input shard is used to pick centers (acknowledged in a TODO).
    # Normalizes the first shard for cosine distance, then calls the kmeans plus plus
    # initialization op for the remaining count with the stored seed and retry count. Returns the
    # chosen centers.
    # Points from only the first shard are used for initializing centers.
    # TODO(ands): Use all points.
    inp = self._inputs[0]
    if self._distance_metric == COSINE_DISTANCE:
      inp = nn_impl.l2_normalize(inp, dim=1)
    return gen_clustering_ops.kmeans_plus_plus_initialization(
        inp, math_ops.cast(self._num_remaining, dtypes.int64), self._seed,
        self._kmeans_plus_plus_num_retries)

  def _kmc2_multiple_centers(self):
    # [FUNCTION DOC]
    # HELPER: k-MC2 initializer.
    # KNOWN ISSUE: the docstring says it returns an op that adds centers, but the returned value
    # is the tensor of centers still remaining after the loop; the center variables are assigned
    # as a side effect inside the loop.
    # Uses only the first shard. Computes how many centers to add this call (at least one, at most
    # the remaining count, at most batch size over chain length) and runs a while loop that adds
    # one center per iteration, assigning the center variable (and updated copy if distinct).
    """Adds new initial cluster centers using the k-MC2 algorithm.

    In each call to the op, the provided batch is split into subsets based on
    the specified `kmc2_chain_length`. On each subset, a single Markov chain of
    the k-MC2 algorithm is used to add *one* new center cluster center. If there
    are less than `kmc2_chain_length` points in the subset, a single center is
    added using one Markov chain on the full input. It is assumed that the
    provided batch has previously been randomly permuted. Otherwise, k-MC2 may
    return suboptimal centers.

    Returns:
      An op that adds new cluster centers.
    """
    # The op only operates on the first shard of data.
    first_shard = self._inputs[0]
    # Number of points in the input that can be used.
    batch_size = array_ops.shape(first_shard)[0]
    # Maximum number of subsets such that the size of each subset is at least
    # `kmc2_chain_length`. Final subsets may be larger.
    max_to_sample = math_ops.cast(
        batch_size / self._kmc2_chain_length, dtype=dtypes.int32)
    # We sample at least one new center and at most all remaining centers.
    num_to_sample = math_ops.maximum(
        math_ops.minimum(self._num_remaining, max_to_sample), 1)

    def _cond(i, _):
      # [FUNCTION DOC]
      # HELPER: loop condition. Returns true while the iteration counter is below the number of
      # centers to sample; the second loop variable is ignored.
      """Stopping condition for the while loop."""
      return math_ops.less(i, num_to_sample)

    def _body(i, _):
      # [FUNCTION DOC]
      # HELPER: loop body. Adds one new center: a random first point if none are selected, else a
      # k-MC2 chain sample from the i-th subset, then assigns the center variable (and updated
      # copy). Returns the next counter and the new remaining-centers count; the incoming second
      # loop variable is ignored.
      """Body that adds a single new center based on a subset."""

      def _sample_random():
        # [FUNCTION DOC]
        # HELPER: branch used when no centers exist. Returns the first data point of the first
        # shard as a one-row matrix (normalized for cosine), relying on the batch being shuffled.
        """Returns a random point as a cluster center."""
        # By assumption the batch is reshuffled and _sample_random is always
        # called for i=0. Hence, we simply return the first point.
        new_center = array_ops.reshape(first_shard[0], [1, -1])
        if self._distance_metric == COSINE_DISTANCE:
          new_center = nn_impl.l2_normalize(new_center, dim=1)
        return new_center

      def _sample_kmc2_chain():
        # [FUNCTION DOC]
        # HELPER: branch used when centers exist. Takes the i-th chain-length slice of the first
        # shard, computes its distances to current centers, samples one index with the k-MC2
        # initialization op, and returns existing centers with the new (normalized for cosine)
        # center appended as a new row.
        """Returns previous centers as well as a new center sampled using k-MC2."""
        # Extract the subset from the underlying batch.
        start = i * self._kmc2_chain_length
        end = start + self._kmc2_chain_length
        subset = first_shard[start:end]
        # Compute the distances from points in the subset to previous centers.
        _, distances = gen_clustering_ops.nearest_neighbors(
            subset, self._cluster_centers, 1)
        # Sample index of new center using k-MC2 Markov chain.
        new_center_index = gen_clustering_ops.kmc2_chain_initialization(
            array_ops.squeeze(distances), self._seed)
        # Extract actual new center.
        newly_sampled_center = array_ops.reshape(subset[new_center_index],
                                                 [1, -1])
        # Return concatenation with previously sampled centers.
        if self._distance_metric == COSINE_DISTANCE:
          newly_sampled_center = nn_impl.l2_normalize(
              newly_sampled_center, dim=1)
        return array_ops.concat([self._cluster_centers, newly_sampled_center],
                                0)

      # Obtain a random point if there are no previously sampled centers.
      # Otherwise, construct a k-MC2 Markov chain.
      new_centers = cond.cond(
          math_ops.equal(self._num_selected, 0), _sample_random,
          _sample_kmc2_chain)
      # Assign new cluster centers to underlying variable.
      assigned_centers = state_ops.assign(
          self._cluster_centers, new_centers, validate_shape=False)
      if self._cluster_centers_updated is not self._cluster_centers:
        assigned_centers = state_ops.assign(
            self._cluster_centers_updated,
            assigned_centers,
            validate_shape=False)
      return i + 1, self._num_clusters - array_ops.shape(assigned_centers)[0]

    # Add num_to_sample new data points.
    _, num_remaining = while_loop.while_loop(_cond, _body, [0, 0])
    return num_remaining

  def _greedy_batch_sampler(self, sampler):
    # [FUNCTION DOC]
    # HELPER: wraps a sampler callback. If the data size is at most the number of remaining
    # centers, returns the concatenation of all inputs; otherwise calls the sampler callback.
    # Returns a conditional tensor.
    # If the input dataset size is smaller than the number of centers
    # remaining, choose the entire input dataset as centers. This can happen
    # with mini-batch. Otherwise, sample the batch according to the provided
    # sampler.
    return cond.cond(self._num_data <= self._num_remaining,
                     lambda: array_ops.concat(self._inputs, 0),
                     sampler)

  def _single_batch_sampler(self, sampler):
    # [FUNCTION DOC]
    # HELPER: wraps a sampler callback with an assertion op requiring the data size to be at least
    # the number of remaining centers (fails at run time otherwise), then returns the sampler
    # callback's result.
    # Enforce that there are at least as many data points as centers
    # remaining. This gives the provided sampler the chance to select all
    # remaining centers from a single batch.
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(self._num_data, self._num_remaining)]):
      return sampler()

  def _choose_initial_centers(self):
    # [FUNCTION DOC]
    # HELPER: dispatches on the initialization choice. Random uses the greedy sampler; any other
    # string is treated as kmeans plus plus with the single-batch sampler; a callable is invoked
    # with the inputs and remaining count; otherwise explicit values are returned after asserting
    # their row count equals the remaining count. Returns the new center rows.
    if isinstance(self._initial_clusters, str):
      if self._initial_clusters == RANDOM_INIT:
        return self._greedy_batch_sampler(self._random)
      else:  # self._initial_clusters == KMEANS_PLUS_PLUS_INIT
        return self._single_batch_sampler(self._kmeans_plus_plus)
    elif callable(self._initial_clusters):
      return self._initial_clusters(self._inputs, self._num_remaining)
    else:
      with ops.control_dependencies([
          check_ops.assert_equal(self._num_remaining,
                                 array_ops.shape(self._initial_clusters)[0])
      ]):
        return self._initial_clusters

  def _add_new_centers(self):
    # [FUNCTION DOC]
    # HELPER: appends newly chosen centers to the center variable.
    # Chooses centers, normalizes them for cosine, concatenates with existing ones (or uses just
    # the new ones when none exist), assigns the center variable (and updated copy if distinct)
    # without shape validation, and returns the number of centers still missing. Duplicates are not
    # removed (see TODO).
    """Adds some centers and returns the number of centers remaining."""
    new_centers = self._choose_initial_centers()
    if self._distance_metric == COSINE_DISTANCE:
      new_centers = nn_impl.l2_normalize(new_centers, dim=1)
    # If cluster_centers is empty, it doesn't have the right shape for concat.
    all_centers = cond.cond(
        math_ops.equal(self._num_selected, 0), lambda: new_centers,
        lambda: array_ops.concat([self._cluster_centers, new_centers], 0))
    # TODO(ccolby): De-dupe all_centers?
    a = state_ops.assign(
        self._cluster_centers, all_centers, validate_shape=False)
    if self._cluster_centers_updated is not self._cluster_centers:
      a = state_ops.assign(
          self._cluster_centers_updated, a, validate_shape=False)
    return self._num_clusters - array_ops.shape(a)[0]

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: initialization branch. Asserts that centers remain to be chosen, then adds centers
    # via the k-MC2 path or the generic path, and returns a conditional op that sets the
    # initialized flag to true when none remain, else a no-op.
    with ops.control_dependencies([
        check_ops.assert_positive(self._num_remaining),
    ]):
      if self._initial_clusters == KMC2_INIT:
        num_now_remaining = self._kmc2_multiple_centers()
      else:
        num_now_remaining = self._add_new_centers()
      return cond.cond(
          math_ops.equal(num_now_remaining, 0),
          lambda: state_ops.assign(self._cluster_centers_initialized, True),
          control_flow_ops.no_op)

  def op(self):
    # [FUNCTION DOC]
    # Public entry point of the factory. Returns a conditional op: if no centers remain it
    # asserts the initialized flag is already true; otherwise it runs the initialization branch.
    """Returns the cluster initializer op."""
    return cond.cond(
        math_ops.equal(self._num_remaining, 0),
        lambda: check_ops.assert_equal(self._cluster_centers_initialized, True),
        self._initialize)
