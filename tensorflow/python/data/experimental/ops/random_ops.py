# [MODULE DOC]
# Experimental random-number dataset. Defines RandomDatasetV2 (a thin subclass of the private
# random dataset from the ops module) and RandomDatasetV1 (adapter wrapping the v2 class for
# TF1 style datasets), both deprecated in favor of the Dataset random method. The unversioned
# name RandomDataset is bound to one of them.
# Import-time side effects: export registration (v2 and v1 endpoints); binds RandomDataset by
# whether TF2 behavior is enabled and registers a callback with the v2 compatibility module to
# rebind it if that setting changes.
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
# ==============================================================================
"""Datasets for random number generators."""
import functools

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import random_op
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# TODO(b/260143413): Migrate users to `tf.data.Dataset.random`.
@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export("data.experimental.RandomDataset", v1=[])
class RandomDatasetV2(random_op._RandomDataset):  # pylint: disable=protected-access
  # [CLASS DOC]
  # Deprecated version 2 endpoint: a dataset of pseudorandom 64-bit integers. Adds no behavior
  # of its own; construction, seeding and element production are inherited from the private
  # random dataset base class.
  """A `Dataset` of pseudorandom values."""


@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export(v1=["data.experimental.RandomDataset"])
class RandomDatasetV1(dataset_ops.DatasetV1Adapter):
  # [CLASS DOC]
  # Deprecated version 1 endpoint: adapter exposing the version 2 random dataset through the
  # v1 dataset interface. Holds the wrapped v2 dataset through the adapter base class.
  """A `Dataset` of pseudorandom values."""

  @functools.wraps(RandomDatasetV2.__init__)
  def __init__(self, seed=None):
    # [FUNCTION DOC]
    # Input: seed, optional random seed (default nothing, meaning the global/random seed
    # behavior of the v2 dataset). Creates the version 2 random dataset with that seed and
    # passes it to the adapter constructor. Returns nothing. The signature documentation is
    # copied from the v2 constructor by a decorator.
    wrapped = RandomDatasetV2(seed)
    super(RandomDatasetV1, self).__init__(wrapped)


if tf2.enabled():
  RandomDataset = RandomDatasetV2
else:
  RandomDataset = RandomDatasetV1


def _tf2_callback():
  # [FUNCTION DOC]
  # HELPER: callback registered with the v2 compatibility module. Rebinds the module-level name
  # RandomDataset to the v2 class when TF2 behavior is enabled, otherwise to the v1 class.
  # Takes no inputs, returns nothing; mutates a module global.
  global RandomDataset
  if tf2.enabled():
    RandomDataset = RandomDatasetV2
  else:
    RandomDataset = RandomDatasetV1


v2_compat.register_data_v2_callback(_tf2_callback)
