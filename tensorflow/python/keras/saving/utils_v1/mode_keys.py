# [MODULE DOC]
# HELPER: defines the standard Keras mode names (train, test, predict) via KerasModeKeys, small
# predicates that test a mode string, and ModeKeyMap, an immutable mapping keyed by modes. Used by
# the export utilities to map modes to MetaGraph tags and signature keys. Mirrors a copy kept in
# the saved_model model_utils package. No import-time side effects.
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
# LINT.IfChange
"""Utils for managing different mode strings used by Keras models.
"""

import collections


class KerasModeKeys:
  # [CLASS DOC]
  # Namespace of string constants naming the three model modes: train, test (evaluation) and
  # predict. Never instantiated; only the class attributes are used.
  """Standard names for model modes.

  The following standard keys are defined:

  * `TRAIN`: training/fitting mode.
  * `TEST`: testing/evaluation mode.
  * `PREDICT`: prediction/inference mode.
  """

  TRAIN = 'train'
  TEST = 'test'
  PREDICT = 'predict'


def is_predict(mode):
  # [FUNCTION DOC]
  # HELPER: returns true if the given mode string equals the predict mode name.
  return mode == KerasModeKeys.PREDICT


def is_eval(mode):
  # [FUNCTION DOC]
  # HELPER: returns true if the given mode string equals the test (evaluation) mode name.
  return mode == KerasModeKeys.TEST


def is_train(mode):
  # [FUNCTION DOC]
  # HELPER: returns true if the given mode string equals the train mode name.
  return mode == KerasModeKeys.TRAIN


class ModeKeyMap(collections.abc.Mapping):
  # [CLASS DOC]
  # Read-only mapping whose keys are mode names. Keys given at construction are remembered in
  # insertion order for iteration, while values are stored under canonical mode names, so lookups
  # accept any key that resolves to the same mode. Used for mapping modes to export tags or
  # signature keys. Invalid mode names are rejected with an error.
  """Map using ModeKeys as keys.

  This class creates an immutable mapping from modes to values. For example,
  SavedModel export of Keras models use this to map modes to their
  corresponding MetaGraph tags/SignatureDef keys.
  """

  def __init__(self, **kwargs):
    # [FUNCTION DOC]
    # Builds the mapping from keyword arguments whose names are mode names and whose values are
    # arbitrary. Records the key order, validates each key and stores the value under its canonical
    # mode. Raises a value error for an invalid mode name or when two keys resolve to the same mode.
    self._internal_dict = {}
    self._keys = []
    for key in kwargs:
      self._keys.append(key)
      dict_key = self._get_internal_key(key)
      if dict_key in self._internal_dict:
        raise ValueError(
            'Error creating ModeKeyMap. Multiple keys/values found for {} mode.'
            .format(dict_key))
      self._internal_dict[dict_key] = kwargs[key]

  def _get_internal_key(self, key):
    # [FUNCTION DOC]
    # HELPER: maps a mode string to its canonical mode name (train, test or predict). Input: key,
    # a string. Raises a value error if it matches none of the modes.
    """Return keys used for the internal dictionary."""
    if is_train(key):
      return KerasModeKeys.TRAIN
    if is_eval(key):
      return KerasModeKeys.TEST
    if is_predict(key):
      return KerasModeKeys.PREDICT
    raise ValueError('Invalid mode key: {}.'.format(key))

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Looks up the value for a mode. Input: key, a mode string. Raises a value error for an invalid
    # mode, or a missing-key error when the mode was not provided at construction.
    return self._internal_dict[self._get_internal_key(key)]

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the keys in the order they were supplied at construction.
    return iter(self._keys)

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of modes in the mapping.
    return len(self._keys)
# LINT.ThenChange(//tensorflow/python/saved_model/model_utils/mode_keys.py)
