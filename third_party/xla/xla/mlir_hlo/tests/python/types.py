# [MODULE DOC]
# Test script for the Python API that accesses MHLO types. Each test function is decorated with a
# runner that creates an MLIR context, registers the mhlo dialect, runs the test and returns it.
# Because the decorator runs the test at import time, executing the script performs all tests.
# Currently it covers the token type: it is constructed and its textual form is checked.
# Copyright 2021 The OpenXLA Authors.
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
"""Test for Python APIs accessing MHLO types."""

# pylint: disable=wildcard-import,undefined-variable,missing-function-docstring

from mlir import ir
from mlir.dialects import mhlo


def run(f):
  # [FUNCTION DOC]
  # HELPER: decorator used by the tests in this file to execute each test inside an MLIR context.
  # Takes a test function f with no arguments. Creates a fresh MLIR context, registers the mhlo
  # dialect in it, calls f while the context is active, then returns f unchanged. Because it is
  # used as a decorator, each test runs once at import time. Exceptions (assertion failures)
  # from f propagate to the caller; the context is released on exit.
  with ir.Context() as context:
    mhlo.register_mhlo_dialect(context)
    f()
  return f


@run
def test_token_type():
  # [FUNCTION DOC]
  # Test: gets the mhlo token type and asserts it is not null and prints as the mhlo token
  # type text. Takes no arguments, returns nothing; fails with an assertion error otherwise.
  token_type = mhlo.TokenType.get()
  assert token_type is not None
  assert str(token_type) == "!mhlo.token"
