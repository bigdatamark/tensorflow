# [MODULE DOC]
# Unit tests for the debugger common module: checks that run keys built from feeds and fetches
# contain the expected flattened tensor names. Tests run only under TF1 behavior because they rely
# on tensor names.
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
"""Unit tests for common values and methods of TensorFlow Debugger."""
import json

from tensorflow.python.debug.lib import common
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class CommonTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the run-key construction function of the debugger common module.

  @test_util.run_v1_only("Relies on tensor name, which is unavailable in TF2")
  def testOnFeedOneFetch(self):
    # [FUNCTION DOC]
    # Builds a run key with one feed and one fetch (two named constants), decodes the JSON and
    # asserts the feed names are the first constant's name and the fetch names the second's.
    a = constant_op.constant(10.0, name="a")
    b = constant_op.constant(20.0, name="b")
    run_key = common.get_run_key({"a": a}, [b])
    loaded = json.loads(run_key)
    self.assertItemsEqual(["a:0"], loaded[0])
    self.assertItemsEqual(["b:0"], loaded[1])

  @test_util.run_v1_only("Relies on tensor name, which is unavailable in TF2")
  def testGetRunKeyFlat(self):
    # [FUNCTION DOC]
    # Builds a run key with one feed and a flat list of two fetches; asserts the decoded feed names
    # and both fetch names (order ignored).
    a = constant_op.constant(10.0, name="a")
    b = constant_op.constant(20.0, name="b")
    run_key = common.get_run_key({"a": a}, [a, b])
    loaded = json.loads(run_key)
    self.assertItemsEqual(["a:0"], loaded[0])
    self.assertItemsEqual(["a:0", "b:0"], loaded[1])

  @test_util.run_v1_only("Relies on tensor name, which is unavailable in TF2")
  def testGetRunKeyNestedFetches(self):
    # [FUNCTION DOC]
    # Builds a run key with empty feeds and fetches nested as a mapping of lists and mappings over
    # four constants; asserts no feed names and that all four names appear flattened in the fetches.
    a = constant_op.constant(10.0, name="a")
    b = constant_op.constant(20.0, name="b")
    c = constant_op.constant(30.0, name="c")
    d = constant_op.constant(30.0, name="d")
    run_key = common.get_run_key(
        {}, {"set1": [a, b], "set2": {"c": c, "d": d}})
    loaded = json.loads(run_key)
    self.assertItemsEqual([], loaded[0])
    self.assertItemsEqual(["a:0", "b:0", "c:0", "d:0"], loaded[1])


if __name__ == "__main__":
  googletest.main()
