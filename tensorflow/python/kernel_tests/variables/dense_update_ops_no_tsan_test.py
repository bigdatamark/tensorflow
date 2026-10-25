# [MODULE DOC]
# Test module for variable assign and assign-add ops under concurrent multi-threaded updates, with
# and without locking. These tests contain deliberate or known data races, so they are excluded
# from thread-sanitizer targets. Defines one test class; no import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for state updating ops that may have benign race conditions."""

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class AssignOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case that launches many threads each running one assign or assign-add op against a
  # shared 1024 by 1024 float variable, checking the outcome bounds for the unlocked and locked
  # variants. No shared fixtures.

  # NOTE(mrry): We exclude thess tests from the TSAN TAP target, because they
  #   contain benign and deliberate data races when multiple threads update
  #   the same parameters without a lock.
  def testParallelUpdateWithoutLocking(self):
    # [FUNCTION DOC]
    # Scenario: switch the default graph to thread-local device stacks, then run 20 assign-add
    # ops (adding ones, no locking) concurrently on 20 threads against a zero variable. Because
    # updates may be lost to races, asserts every element lies between 1 and 20 inclusive.
    # Side effect: modifies the default graph's thread-local setting.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      ones_t = array_ops.fill([1024, 1024], 1.0)
      p = variables.Variable(array_ops.zeros([1024, 1024]))
      adds = [
          state_ops.assign_add(
              p, ones_t, use_locking=False) for _ in range(20)
      ]
      self.evaluate(variables.global_variables_initializer())

      def run_add(add_op):
        # [FUNCTION DOC]
        # HELPER: Thread target for the parallel update test. Evaluates the given op in the
        # test session; returns nothing.
        self.evaluate(add_op)

      threads = [
          self.checkedThread(
              target=run_add, args=(add_op,)) for add_op in adds
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = self.evaluate(p)
      ones = np.ones((1024, 1024)).astype(np.float32)
      self.assertTrue((vals >= ones).all())
      self.assertTrue((vals <= ones * 20).all())

  def testParallelAssignWithoutLocking(self):
    # [FUNCTION DOC]
    # Scenario: with thread-local graph state, run 20 unlocked assign ops concurrently, the i-th
    # assigning a matrix filled with i (1 to 20), to a zero variable. Asserts every element ends
    # up greater than zero and at most 20. Side effect: sets the default graph thread-local.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      ones_t = array_ops.fill([1024, 1024], float(1))
      p = variables.Variable(array_ops.zeros([1024, 1024]))
      assigns = [
          state_ops.assign(p, math_ops.multiply(ones_t, float(i)), False)
          for i in range(1, 21)
      ]
      self.evaluate(variables.global_variables_initializer())

      def run_assign(assign_op):
        # [FUNCTION DOC]
        # HELPER: Thread target for the parallel assign test. Evaluates the given assign op in
        # the test session; returns nothing.
        self.evaluate(assign_op)

      threads = [
          self.checkedThread(
              target=run_assign, args=(assign_op,)) for assign_op in assigns
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = self.evaluate(p)

      # Assert every element is taken from one of the assignments.
      self.assertTrue((vals > 0).all())
      self.assertTrue((vals <= 20).all())

  # NOTE(skyewm): We exclude these tests from the TSAN TAP target, because they
  # contain non-benign but known data races between the variable assignment and
  # returning the output tensors. This issue will be resolved with the new
  # resource variables.
  def testParallelUpdateWithLocking(self):
    # [FUNCTION DOC]
    # Scenario: with thread-local graph state, run 20 assign-add ops of ones with locking enabled
    # concurrently on a zero variable. Because updates are serialized, asserts every element
    # equals exactly 20. Side effect: sets the default graph thread-local.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      zeros_t = array_ops.fill([1024, 1024], 0.0)
      ones_t = array_ops.fill([1024, 1024], 1.0)
      p = variables.Variable(zeros_t)
      adds = [
          state_ops.assign_add(
              p, ones_t, use_locking=True) for _ in range(20)
      ]
      self.evaluate(p.initializer)

      def run_add(add_op):
        # [FUNCTION DOC]
        # HELPER: Thread target for the locked parallel update test. Evaluates the given op in
        # the test session; returns nothing.
        self.evaluate(add_op)

      threads = [
          self.checkedThread(
              target=run_add, args=(add_op,)) for add_op in adds
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = self.evaluate(p)
      ones = np.ones((1024, 1024)).astype(np.float32)
      self.assertAllEqual(vals, ones * 20)

  def testParallelAssignWithLocking(self):
    # [FUNCTION DOC]
    # Scenario: with thread-local graph state, run 20 locked assign ops (the i-th assigning a
    # matrix filled with i) concurrently on a zero variable. Asserts the first element is between
    # 1 and 20 and that all elements equal that first element, i.e. one assignment won as a whole.
    # Side effect: sets the default graph thread-local.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      zeros_t = array_ops.fill([1024, 1024], 0.0)
      ones_t = array_ops.fill([1024, 1024], 1.0)
      p = variables.Variable(zeros_t)
      assigns = [
          state_ops.assign(
              p, math_ops.multiply(ones_t, float(i)), use_locking=True)
          for i in range(1, 21)
      ]
      self.evaluate(p.initializer)

      def run_assign(assign_op):
        # [FUNCTION DOC]
        # HELPER: Thread target for the locked parallel assign test. Evaluates the given assign
        # op in the test session; returns nothing.
        self.evaluate(assign_op)

      threads = [
          self.checkedThread(
              target=run_assign, args=(assign_op,)) for assign_op in assigns
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = self.evaluate(p)

      # Assert every element is the same, and taken from one of the assignments.
      self.assertTrue(vals[0, 0] > 0)
      self.assertTrue(vals[0, 0] <= 20)
      self.assertAllEqual(vals, np.ones([1024, 1024]) * vals[0, 0])


if __name__ == "__main__":
  test.main()
