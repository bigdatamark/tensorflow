# [MODULE DOC]
# Unit tests for the C API utility module: the API definition map wrapper (listing op names,
# fetching op and API definitions, overriding an API definition from text) and the
# single-ownership pointer wrapper (deletion on collection, aliveness inside the access context,
# error after deletion). Runs through the standard test runner when executed as a script.
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

"""Tests for c_api utils."""
import gc
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ApiDefMapTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the API definition map wrapper: op name listing, op and API definition lookup for
  # the Add op, and overriding an API definition from text. No shared fixtures.

  def testApiDefMapOpNames(self):
    # [FUNCTION DOC]
    # Scenario: create an API definition map and list its op names. Asserts that the Add op is
    # among them.
    api_def_map = c_api_util.ApiDefMap()
    self.assertIn("Add", api_def_map.op_names())

  def testApiDefMapGet(self):
    # [FUNCTION DOC]
    # Scenario: create an API definition map and fetch both the op definition and API definition
    # for the Add op. Asserts the op definition name and the API definition graph op name are Add.
    api_def_map = c_api_util.ApiDefMap()
    op_def = api_def_map.get_op_def("Add")
    self.assertEqual(op_def.name, "Add")
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")

  def testApiDefMapPutThenGet(self):
    # [FUNCTION DOC]
    # Scenario: create an API definition map, put a text API definition overriding the summary and
    # description of the Add op, then fetch it. Asserts the graph op name is Add and the summary
    # equals the overridden text.
    api_def_map = c_api_util.ApiDefMap()
    api_def_text = """
op {
  graph_op_name: "Add"
  summary: "Returns x + y element-wise."
  description: <<END
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
END
}
"""
    api_def_map.put_api_def(api_def_text)
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")
    self.assertEqual(api_def.summary, "Returns x + y element-wise.")


class UniquePtrTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the single-ownership pointer wrapper. The shared fixture creates a mock object that
  # records deletion and a deleter callback; tests cover normal life cycle, safety when the wrapper
  # reference is dropped during use, and the error raised after deletion.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setup, defines a mock class whose instances carry a deleted flag
    # (initially false) and a deleter function that sets that flag, and stores a fresh mock
    # instance and the deleter on the test case for the tests to wrap.

    super(UniquePtrTest, self).setUp()

    class MockClass:
      # [CLASS DOC]
      # Minimal stand-in for a native object; holds a boolean deleted flag set by the deleter.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor: sets the deleted flag to false.
        self.deleted = False

    def deleter(obj):
      # [FUNCTION DOC]
      # Deleter callback given to the wrapper: marks the given mock object as deleted.
      obj.deleted = True

    self.obj = MockClass()
    self.deleter = deleter

  def testLifeCycle(self):
    # [FUNCTION DOC]
    # Scenario: wrap the mock object with a deleter. Asserts it is not deleted initially, that the
    # access context yields the same object, and that after dropping the wrapper and collecting
    # garbage the deleter has run and marked the object deleted.
    self.assertFalse(self.obj.deleted)

    a = c_api_util.UniquePtr(name="mock", deleter=self.deleter, obj=self.obj)

    with a.get() as obj:
      self.assertIs(obj, self.obj)

    del a
    gc.collect()
    self.assertTrue(self.obj.deleted)

  def testSafeUnderRaceCondition(self):
    # [FUNCTION DOC]
    # Scenario: the only reference to the wrapper is dropped while inside its access context,
    # mimicking another thread releasing it. Asserts the wrapped object is not deleted while the
    # context is active, and that it is deleted once the context exits and collection runs.
    self.assertFalse(self.obj.deleted)

    a = c_api_util.UniquePtr(name="mock", deleter=self.deleter, obj=self.obj)

    with a.get() as obj:
      self.assertIs(obj, self.obj)
      # The del below mimics a potential race condition.
      # 'a' could be owned by a different thread, and this thread not
      # necessarily hold a long-term reference to a.
      del a
      gc.collect()
      self.assertFalse(obj.deleted)

    gc.collect()
    self.assertTrue(self.obj.deleted)

  def testRaiseAfterDeleted(self):
    # [FUNCTION DOC]
    # Scenario: a wrapper is explicitly finalized (mimicking a deletion started elsewhere).
    # Asserts the mock object is marked deleted, then that entering the access context raises the
    # already-garbage-collected error whose message mentions the mock class name, and that the
    # object stays deleted after a collection.
    self.assertFalse(self.obj.deleted)

    a = c_api_util.UniquePtr(name="mock", deleter=self.deleter, obj=self.obj)

    # The __del__ below mimics a partially started deletion, potentially
    # started from another thread.
    # 'a' could be owned by a different thread, and this thread not
    # necessarily hold a long-term reference to a.
    a.__del__()
    self.assertTrue(self.obj.deleted)

    with self.assertRaisesRegex(c_api_util.AlreadyGarbageCollectedError,
                                "MockClass"):
      with a.get():
        pass

    gc.collect()
    self.assertTrue(self.obj.deleted)

if __name__ == "__main__":
  googletest.main()
