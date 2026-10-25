# [MODULE DOC]
# Unit tests for tensorflow.python.trackable.resource. Covers ResourceTracker with its
# scope manager (single, sequential and nested scopes) and the resource creator hooks that
# can rewrite constructor arguments or post-process resources. Defines small dummy resource
# classes and a decorator helper to run graph-style code in eager mode. Run through the
# standard TensorFlow test main entry point.
#
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

from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.trackable import resource


def run_inside_wrap_function_in_eager_mode(graph_function):
  # [FUNCTION DOC]
  # HELPER: test decorator letting graph-style test bodies run in both eager and graph modes.
  # Input: graph_function, a test method taking the test case. Returns a replacement method
  # taking the test case: in eager mode it wraps the original with the function wrapper
  # utility (passing the test case as argument) and calls the wrapped function; in graph mode
  # it just calls the original.
  """Decorator to execute the same graph code in eager and graph modes.

  In graph mode, we just execute the graph_function passed as argument. In eager
  mode, we wrap the function using wrap_function and then execute the wrapped
  result.

  Args:
    graph_function: python function containing graph code to be wrapped

  Returns:
    decorated function
  """
  def wrap_and_execute(self):
    # [FUNCTION DOC]
    # HELPER: the replacement test method built by the decorator. Input: self, the test case.
    # Chooses eager wrapping or direct execution as described for the decorator. Returns
    # nothing.
    if context.executing_eagerly():
      wrapped = wrap_function.wrap_function(graph_function, [self])
      # use the wrapped graph function
      wrapped()
    else:
      # use the original function
      graph_function(self)
  return wrap_and_execute


class _DummyResource(resource.TrackableResource):
  # [CLASS DOC]
  # HELPER: minimal test resource whose "handle" is simply a name string given at
  # construction. Holds that name; used to test trackers and creator hooks.

  def __init__(self, handle_name):
    # [FUNCTION DOC]
    # HELPER: constructor. Input: handle_name, a string stored and later returned as the
    # handle. Stores it, then runs the parent constructor (which registers with trackers).
    self._handle_name = handle_name
    super(_DummyResource, self).__init__()

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: returns the stored name string in place of a real handle tensor.
    return self._handle_name


class _DummyResource1(resource.TrackableResource):
  # [CLASS DOC]
  # HELPER: like the dummy resource but also holds an integer value attribute starting at
  # zero, so tests can detect a creator hook that modified the created instance.

  def __init__(self, handle_name):
    # [FUNCTION DOC]
    # HELPER: constructor. Input: handle_name, a string. Stores it, sets the value attribute
    # to zero, then runs the parent constructor.
    self._handle_name = handle_name
    self._value = 0
    super(_DummyResource1, self).__init__()

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: returns the stored name string in place of a real handle tensor.
    return self._handle_name


class ResourceTrackerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for ResourceTracker and resource_tracker_scope: checks which trackers collect
  # which resources for single, consecutive and nested scopes. No shared fixtures.

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: two dummy resources are created inside one tracker scope. Expects the tracker
    # to hold exactly two resources, in creation order, with handles test1 then test2.
    resource_tracker = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker):
      dummy_resource1 = _DummyResource("test1")
      dummy_resource2 = _DummyResource("test2")

    self.assertEqual(2, len(resource_tracker.resources))
    self.assertEqual("test1", resource_tracker.resources[0].resource_handle)
    self.assertEqual("test2", resource_tracker.resources[1].resource_handle)

  def testTwoScopes(self):
    # [FUNCTION DOC]
    # Scenario: two separate consecutive scopes each create one resource. Expects each tracker
    # to contain only the resource made in its own scope, with the matching handle name.
    resource_tracker1 = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker1):
      dummy_resource1 = _DummyResource("test1")

    resource_tracker2 = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker2):
      dummy_resource2 = _DummyResource("test2")

    self.assertEqual(1, len(resource_tracker1.resources))
    self.assertEqual("test1", resource_tracker1.resources[0].resource_handle)
    self.assertEqual(1, len(resource_tracker2.resources))
    self.assertEqual("test2", resource_tracker2.resources[0].resource_handle)

  def testNestedScopesScopes(self):
    # [FUNCTION DOC]
    # Scenario: an outer scope contains two inner scopes, each creating one resource. Expects
    # each inner tracker to hold only its own resource and the outer tracker to hold both, in
    # creation order.
    resource_tracker = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker):
      resource_tracker1 = resource.ResourceTracker()
      with resource.resource_tracker_scope(resource_tracker1):
        dummy_resource1 = _DummyResource("test1")

      resource_tracker2 = resource.ResourceTracker()
      with resource.resource_tracker_scope(resource_tracker2):
        dummy_resource2 = _DummyResource("test2")

    self.assertEqual(1, len(resource_tracker1.resources))
    self.assertEqual("test1", resource_tracker1.resources[0].resource_handle)
    self.assertEqual(1, len(resource_tracker2.resources))
    self.assertEqual("test2", resource_tracker2.resources[0].resource_handle)
    self.assertEqual(2, len(resource_tracker.resources))
    self.assertEqual("test1", resource_tracker.resources[0].resource_handle)
    self.assertEqual("test2", resource_tracker.resources[1].resource_handle)


class ResourceCreatorScopeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the graph-level resource creator scope mechanism that intercepts resource
  # construction (via the resource metaclass). Tests run in both graph and eager modes, using
  # the wrap-function decorator helper in eager mode.

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testResourceCreator(self):
    # [FUNCTION DOC]
    # Scenario: one creator hook that forces the handle name argument is registered for both
    # dummy resource types. Two resources are built with different names inside the scope.
    # Expects both to have the forced name, showing one hook serves several classes.
    def resource_creator_fn(next_creator, *a, **kwargs):
      # [FUNCTION DOC]
      # HELPER: test creator hook that overrides the handle name keyword argument with a fixed
      # string and delegates to the next creator, returning the created instance.
      kwargs["handle_name"] = "forced_name"
      return next_creator(*a, **kwargs)

    # test that two resource classes use the same creator function
    with ops.resource_creator_scope(["_DummyResource", "_DummyResource1"],
                                    resource_creator_fn):
      dummy_0 = _DummyResource(handle_name="fake_name_0")
      dummy_1 = _DummyResource1(handle_name="fake_name_1")

    self.assertEqual(dummy_0._handle_name, "forced_name")
    self.assertEqual(dummy_1._handle_name, "forced_name")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testResourceCreatorNestingError(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the creator scope is called with the creator function and the type name in
    # swapped order relative to the other tests (which pass the names first), so the intended
    # scenario may not actually be exercised as written.
    # Scenario: a scope is entered manually, a second scope is entered inside it, and then the
    # first scope is exited out of order. Expects a runtime error. The original per-type
    # creator stack on the default graph is saved first and restored in a finally clause.

    def creator(next_creator, *a, **kwargs):
      # [FUNCTION DOC]
      # HELPER: pass-through creator hook; calls the next creator with unchanged arguments
      # and returns its result.
      return next_creator(*a, **kwargs)

    # Save the state so we can clean up at the end.
    graph = ops.get_default_graph()
    old_creator_stack = graph._resource_creator_stack["_DummyResource"]

    try:
      scope = ops.resource_creator_scope(creator, "_DummyResource")
      scope.__enter__()
      with ops.resource_creator_scope(creator, "_DummyResource"):
        with self.assertRaises(RuntimeError):
          scope.__exit__(None, None, None)
    finally:
      graph._resource_creator_stack["_DummyResource"] = old_creator_stack

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testResourceCreatorNesting(self):
    # [FUNCTION DOC]
    # Scenario: two creator hooks are nested for the second dummy type; the outer sets the
    # value attribute on the created instance, the inner forces the handle name. One resource
    # is created inside. Expects the forced name and a value of one, showing both hooks ran.

    def resource_creator_fn_0(next_creator, *a, **kwargs):
      # [FUNCTION DOC]
      # HELPER: outer test hook; creates the instance via the next creator, sets its value
      # attribute to one and returns it.
      instance = next_creator(*a, **kwargs)
      instance._value = 1
      return instance

    def resource_creator_fn_1(next_creator, *a, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner test hook; overrides the handle name keyword argument with a fixed
      # string and delegates to the next creator.
      kwargs["handle_name"] = "forced_name1"
      return next_creator(*a, **kwargs)

    with ops.resource_creator_scope(["_DummyResource1"], resource_creator_fn_0):
      with ops.resource_creator_scope(["_DummyResource1"],
                                      resource_creator_fn_1):
        dummy_0 = _DummyResource1(handle_name="fake_name")

    self.assertEqual(dummy_0._handle_name, "forced_name1")
    self.assertEqual(dummy_0._value, 1)


if __name__ == "__main__":
  test.main()
