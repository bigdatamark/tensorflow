# [MODULE DOC]
# Customizes the generic test_combinations machinery for tf.keras tests. It defines the list of
# Keras model types, helpers that build the default parameter combinations (graph/eager mode and
# run-eagerly settings, model types), two test-combination classes that wrap each test in the
# matching testing_utils scope, and re-exports a keras-aware generate decorator plus combine,
# times and NamedObject from the framework modules. At import time it reads the default
# combinations from the framework generate decorator and builds the keras-specific generate
# partial. Used as a helper module by Keras unit tests.
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
"""This module customizes `test_combinations` for `tf.keras` related tests."""

import functools

from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils

KERAS_MODEL_TYPES = ['functional', 'subclass', 'sequential']


def keras_mode_combinations(mode=None, run_eagerly=None):
  # [FUNCTION DOC]
  # HELPER: builds the default parameter combinations used by Keras parameterized tests.
  # Inputs: mode, an optional list of 'graph' and/or 'eager' (default: only eager when TF2 is
  # enabled, otherwise both; an empty list yields no combinations); run_eagerly, an optional list
  # of booleans (default: true and false). Returns a list of combinations, with one entry per
  # run_eagerly value for eager mode and a single entry with run_eagerly false for graph mode.
  # Raises nothing itself and has no side effects beyond reading whether TF2 behavior is on.
  """Returns the default test combinations for tf.keras tests.

  Note that if tf2 is enabled, then v1 session test will be skipped.

  Args:
    mode: List of modes to run the tests. The valid options are 'graph' and
      'eager'. Default to ['graph', 'eager'] if not specified. If a empty list
      is provide, then the test will run under the context based on tf's
      version, eg graph for v1 and eager for v2.
    run_eagerly: List of `run_eagerly` value to be run with the tests.
      Default to [True, False] if not specified. Note that for `graph` mode,
      run_eagerly value will only be False.

  Returns:
    A list contains all the combinations to be used to generate test cases.
  """
  if mode is None:
    mode = ['eager'] if tf2.enabled() else ['graph', 'eager']
  if run_eagerly is None:
    run_eagerly = [True, False]
  result = []
  if 'eager' in mode:
    result += combinations.combine(mode=['eager'], run_eagerly=run_eagerly)
  if 'graph' in mode:
    result += combinations.combine(mode=['graph'], run_eagerly=[False])
  return result


def keras_model_type_combinations():
  # [FUNCTION DOC]
  # HELPER: returns one combination per Keras model type (functional, subclass, sequential) for
  # parameterizing model tests. Takes no inputs and has no side effects.
  return combinations.combine(model_type=KERAS_MODEL_TYPES)


class KerasModeCombination(test_combinations.TestCombination):
  # [CLASS DOC]
  # Test combination that handles the optional run_eagerly test parameter by entering the
  # run-eagerly scope from testing_utils around the test. Stateless; instantiated once and added
  # to the default combinations used by the keras generate decorator.
  """Combination for Keras test mode.

  It by default includes v1_session, v2_eager and v2_tf_function.
  """

  def context_managers(self, kwargs):
    # [FUNCTION DOC]
    # Removes the run_eagerly entry from the test keyword arguments (mutating the mapping) and,
    # if it was given and not null, returns a one-element list with the run-eagerly scope for that
    # value; otherwise returns an empty list.
    run_eagerly = kwargs.pop('run_eagerly', None)

    if run_eagerly is not None:
      return [testing_utils.run_eagerly_scope(run_eagerly)]
    else:
      return []

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Declares that the run_eagerly test parameter is optional, so tests that omit it still run.
    # Returns a one-element list holding that optional-parameter modifier.
    return [test_combinations.OptionalParameter('run_eagerly')]


class KerasModelTypeCombination(test_combinations.TestCombination):
  # [CLASS DOC]
  # Test combination that handles the optional model_type test parameter by entering the
  # model-type scope from testing_utils, so that helpers creating Keras models build the requested
  # kind (functional, subclass or sequential). Stateless; added to the default combinations.
  """Combination for Keras model types when doing model test.

  It by default includes 'functional', 'subclass', 'sequential'.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.
  """

  def context_managers(self, kwargs):
    # [FUNCTION DOC]
    # Removes the model_type entry from the test keyword arguments (mutating the mapping) and, if
    # it is one of the known Keras model types, returns a one-element list with the model-type
    # scope for it; otherwise returns an empty list.
    model_type = kwargs.pop('model_type', None)
    if model_type in KERAS_MODEL_TYPES:
      return [testing_utils.model_type_scope(model_type)]
    else:
      return []

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Declares that the model_type test parameter is optional, so tests that omit it still run.
    # Returns a one-element list holding that optional-parameter modifier.
    return [test_combinations.OptionalParameter('model_type')]


_defaults = combinations.generate.keywords['test_combinations']
generate = functools.partial(
    combinations.generate,
    test_combinations=_defaults +
    (KerasModeCombination(), KerasModelTypeCombination()))
combine = test_combinations.combine
times = test_combinations.times
NamedObject = test_combinations.NamedObject
