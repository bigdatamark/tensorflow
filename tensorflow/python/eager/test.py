# [MODULE DOC]
# HELPER: small shared test utility module for eager-mode tests.
# Re-exports everything from the platform test module (test case base classes, benchmark base,
# and so on) via a wildcard import and overrides the main entry point so that eager execution is
# enabled before the tests are run. Marked by its authors as a candidate for removal.
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
"""Utilities for testing tfe code."""

from tensorflow.python.framework import ops as _ops
from tensorflow.python.platform import test as _test
from tensorflow.python.platform.test import *  # pylint: disable=wildcard-import


# TODO(akshayka): Do away with this file.
def main(argv=None):  # pylint: disable=function-redefined
  # [FUNCTION DOC]
  # HELPER: test runner entry point for eager tests; shadows the wildcard-imported main function.
  # Enables eager execution globally and then delegates to the platform test runner, passing along
  # the optional argument list (default none, meaning the process arguments). Does not return
  # until tests finish; side effect: turns on eager execution for the process.
  _ops.enable_eager_execution()
  _test.main(argv)
