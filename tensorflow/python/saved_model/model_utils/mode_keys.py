# [MODULE DOC]
# Mode-key definitions shared by Keras and Estimator export code: two classes of standard mode
# name strings (Keras uses train/test/predict, Estimator uses train/eval/infer), predicates that
# recognize a mode in either naming scheme, and an immutable mapping keyed by mode that treats
# the equivalent Keras and Estimator names as the same key. Used by export_utils to map modes to
# tags and signature keys. A copy is kept in sync with another module via the lint markers.
# No import-time side effects beyond class definitions.
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
"""Utils for managing different mode strings used by Keras and Estimator models.
"""

from tensorflow.python.util.compat import collections_abc


class KerasModeKeys(object):
  # [CLASS DOC]
  # Namespace of the standard mode strings used by Keras: train, test and predict. Holds only
  # class-level string constants and is never instantiated.
  """Standard names for model modes.

  The following standard keys are defined:

  * `TRAIN`: training/fitting mode.
  * `TEST`: testing/evaluation mode.
  * `PREDICT`: prediction/inference mode.
  """

  TRAIN = 'train'
  TEST = 'test'
  PREDICT = 'predict'


# TODO(kathywu): Remove copy in Estimator after nightlies
class EstimatorModeKeys(object):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring has a typo ("predication") and describes the keys loosely; note
  # that the predict constant's value is the string infer, not predict.
  # Namespace of the standard mode strings used by Estimator: train, eval and infer. Holds only
  # class-level string constants and is never instantiated. A temporary copy of the Estimator one.
  """Standard names for Estimator model modes.

  The following standard keys are defined:

  * `TRAIN`: training/fitting mode.
  * `EVAL`: testing/evaluation mode.
  * `PREDICT`: predication/inference mode.
  """

  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'infer'


def is_predict(mode):
  # [FUNCTION DOC]
  # Returns true if the given mode string equals the Keras predict name or the Estimator
  # infer name; false otherwise. No exceptions or side effects.
  return mode in [KerasModeKeys.PREDICT, EstimatorModeKeys.PREDICT]


def is_eval(mode):
  # [FUNCTION DOC]
  # Returns true if the given mode string equals the Keras test name or the Estimator eval
  # name; false otherwise. No exceptions or side effects.
  return mode in [KerasModeKeys.TEST, EstimatorModeKeys.EVAL]


def is_train(mode):
  # [FUNCTION DOC]
  # Returns true if the given mode string equals the shared Keras/Estimator train name; false
  # otherwise. No exceptions or side effects.
  return mode in [KerasModeKeys.TRAIN, EstimatorModeKeys.TRAIN]


class ModeKeyMap(collections_abc.Mapping):
  # [CLASS DOC]
  # Immutable read-only mapping from mode names to arbitrary values. Keys are normalized to the
  # Keras mode names for lookup so equivalent Keras and Estimator names (predict/infer, test/eval)
  # reach the same value. State: an internal dictionary keyed by normalized mode and a list of the
  # keys exactly as supplied at construction (used for iteration and length). Created from keyword
  # arguments; subclasses the abstract mapping base, implementing item access, iteration, length.
  """Map using ModeKeys as keys.

  This class creates an immutable mapping from modes to values. For example,
  SavedModel export of Keras and Estimator models use this to map modes to their
  corresponding MetaGraph tags/SignatureDef keys.

  Since this class uses modes, rather than strings, as keys, both "predict"
  (Keras's PREDICT ModeKey) and "infer" (Estimator's PREDICT ModeKey) map to the
  same value.
  """

  def __init__(self, **kwargs):
    # [FUNCTION DOC]
    # Builds the mapping from keyword arguments whose names are mode strings and whose values are
    # the mapped values. Records the supplied keys in order, normalizes each and stores the value.
    # Raises a value error if a key is not a valid mode, or if two keys normalize to the same mode
    # (for example both predict and infer). No other side effects.
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
    # HELPER: key normalizer for the mode-keyed mapping.
    # Maps a train, eval/test or predict/infer mode string to the corresponding Keras mode name
    # and returns it. Raises a value error for any other string.
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
    # Looks up a value by mode, treating equivalent Keras and Estimator names as the same key.
    # Raises a value error for an unrecognized mode string and a key error when the mode is valid
    # but absent from the mapping.
    return self._internal_dict[self._get_internal_key(key)]

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the keys exactly as supplied at construction (not normalized),
    # in the order given.
    return iter(self._keys)

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of keys supplied at construction, which equals the number of stored
    # entries because duplicate normalized keys are rejected.
    return len(self._keys)
# LINT.ThenChange(//tensorflow/python/keras/saving/utils_v1/mode_keys.py)
