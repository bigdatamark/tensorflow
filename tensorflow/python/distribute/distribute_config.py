# [MODULE DOC]
# Defines the DistributeConfig named tuple, a small configuration record holding the training
# strategy, an optional evaluation strategy and optional remote cluster settings for high-level
# training APIs. No import-time side effects.

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
"""A configure tuple for high-level APIs for running distribution strategies."""

import collections


class DistributeConfig(
    collections.namedtuple(
        'DistributeConfig',
        ['train_distribute', 'eval_distribute', 'remote_cluster'])):
  # [CLASS DOC]
  # Immutable record with three fields: train_distribute (strategy for training),
  # eval_distribute (strategy for evaluation) and remote_cluster (cluster description as a
  # mapping, cluster definition or cluster spec). Built by keyword or position; all default to
  # nothing. Extends a named tuple type created inline.
  """A config tuple for distribution strategies.

  Attributes:
    train_distribute: a `DistributionStrategy` object for training.
    eval_distribute: an optional `DistributionStrategy` object for
      evaluation.
    remote_cluster: a dict, `ClusterDef` or `ClusterSpec` object specifying
      the cluster configurations. If this is given, the `train_and_evaluate`
      method will be running as a standalone client which connects to the
      cluster for training.
  """

  def __new__(cls,
              train_distribute=None,
              eval_distribute=None,
              remote_cluster=None):
    # [FUNCTION DOC]
    # Constructor: creates the tuple from train_distribute, eval_distribute and remote_cluster,
    # each optional and defaulting to nothing. Returns the new config instance; no validation.
    return super(DistributeConfig, cls).__new__(cls, train_distribute,
                                                eval_distribute, remote_cluster)
