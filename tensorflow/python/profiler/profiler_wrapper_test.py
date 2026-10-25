# [MODULE DOC]
# Smoke tests for the native profiler plugin wrapper: converting an empty list of profile space
# files to trace viewer tool data with no, empty, integer and string options. Only checks that
# the calls do not raise.
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
"""Tests for profiler_wrapper.cc pybind methods."""

from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.profiler.internal import _pywrap_profiler_plugin as profiler_wrapper_plugin


class ProfilerSessionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the native space-to-tools-data conversion binding; no shared fixtures.

  def test_xspace_to_tools_data_default_options(self):
    # [FUNCTION DOC]
    # Calls the conversion with an empty file list and the trace viewer tool, without options.
    # Passes if nothing is raised.
    # filenames only used for `hlo_proto` tool.
    profiler_wrapper_plugin.xspace_to_tools_data([], 'trace_viewer')

  def _test_xspace_to_tools_data_options(self, options):
    # [FUNCTION DOC]
    # HELPER: Shared body for the options tests below.
    # Calls the conversion with an empty file list, the trace viewer tool and the given options
    # mapping. Returns nothing; the test passes if no exception is raised.
    profiler_wrapper_plugin.xspace_to_tools_data([], 'trace_viewer', options)

  def test_xspace_to_tools_data_empty_options(self):
    # [FUNCTION DOC]
    # Runs the options helper with an empty options mapping; passes if nothing is raised.
    self._test_xspace_to_tools_data_options({})

  def test_xspace_to_tools_data_int_options(self):
    # [FUNCTION DOC]
    # Runs the options helper with a mapping holding an integer value; passes if nothing is raised.
    self._test_xspace_to_tools_data_options({'example_option': 0})

  def test_xspace_to_tools_data_str_options(self):
    # [FUNCTION DOC]
    # Runs the options helper with a mapping holding a string value; passes if nothing is raised.
    self._test_xspace_to_tools_data_options({'example_option': 'example'})

if __name__ == '__main__':
  test.main()
