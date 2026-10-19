# [MODULE DOC]
# Test combination definitions specialised for TensorFlow. Provides a combination that runs a
# test in eager or graph mode and one that skips tests according to whether TF2 behavior is
# enabled, and exposes the pre-configured generate, combine, times and named-object helpers built
# on the generic test combinations module. Import-time side effect: exports the generate helper
# in the internal test combinations API namespace.
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
"""This module customizes `test_combinations` for Tensorflow.

Additionally it provides `generate()`, `combine()` and `times()` with Tensorflow
customizations as a default.
"""

import functools

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations
from tensorflow.python.util.tf_export import tf_export


class EagerGraphCombination(test_combinations.TestCombination):
  # [CLASS DOC]
  # Test combination that runs a test body in eager or graph execution mode according to an
  # optional mode parameter. Stateless; instantiated once for the module-level generate helper.
  # Overrides the context manager and parameter modifier hooks of the base combination.
  """Run the test in Graph or Eager mode.

  The optional `mode` parameter controls the test's execution mode.  Its
  accepted values are "graph" or "eager" literals.
  """

  def context_managers(self, kwargs):
    # [FUNCTION DOC]
    # Builds the context managers a test runs inside. Input kwargs is the mapping of test
    # parameters; the mode entry is removed from it (mutation). No mode gives an empty list;
    # "eager" gives a list with the eager mode context; "graph" gives a list with a fresh graph
    # as default plus the graph mode context. Any other value raises a value error naming the
    # received value.
    mode = kwargs.pop("mode", None)
    if mode is None:
      return []
    elif mode == "eager":
      return [context.eager_mode()]
    elif mode == "graph":
      return [ops.Graph().as_default(), context.graph_mode()]
    else:
      raise ValueError(
          "Argument 'mode' must be either 'eager' or 'graph'. "
          f"Received: {mode}.")

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Declares that the mode test parameter is optional. Returns a list with one optional-
    # parameter modifier. No side effects.
    return [test_combinations.OptionalParameter("mode")]


class TFVersionCombination(test_combinations.TestCombination):
  # [CLASS DOC]
  # Test combination that gates tests on the TensorFlow API version in effect. It reads an
  # optional version parameter (1 or 2) from the test arguments and skips the test when it does
  # not match whether TF2 behavior is enabled. Stateless; instantiated once for the module-level
  # generate helper.
  """Control the execution of the test in TF1.x and TF2.

  If TF2 is enabled then a test with TF1 test is going to be skipped and vice
  versa.

  Test targets continuously run in TF2 thanks to the tensorflow.v2 TAP target.
  A test can be run in TF2 with bazel by passing --test_env=TF2_BEHAVIOR=1.
  """

  def should_execute_combination(self, kwargs):
    # [FUNCTION DOC]
    # Decides whether a generated test combination should run. Input kwargs is the mapping of
    # test parameters; the tf_api_version entry is removed from it (mutation). Returns a pair of
    # (should run, skip reason): not run with a message if the version is 1 while TF2 behavior is
    # enabled, or the version is 2 while it is disabled; otherwise run with no reason (also when
    # the parameter is absent).
    tf_api_version = kwargs.pop("tf_api_version", None)
    if tf_api_version == 1 and tf2.enabled():
      return (False, "Skipping a TF1.x test when TF2 is enabled.")
    elif tf_api_version == 2 and not tf2.enabled():
      return (False, "Skipping a TF2 test when TF2 is not enabled.")
    return (True, None)

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Declares that the tf_api_version test parameter is optional, so tests that omit it are
    # still valid. Returns a list holding one optional-parameter modifier. No side effects.
    return [test_combinations.OptionalParameter("tf_api_version")]


generate = functools.partial(
    test_combinations.generate,
    test_combinations=(EagerGraphCombination(), TFVersionCombination()))
combine = test_combinations.combine
times = test_combinations.times
NamedObject = test_combinations.NamedObject

tf_export("__internal__.test.combinations.generate", v1=[])(generate)
