# [MODULE DOC]
# Unit tests for the training_util module (global step helpers). Contains two test classes:
# one covering creation, lookup and validation of the global step variable, and one covering
# the cached global-step read tensor and the increment helper. No import-time side effects
# beyond importing test dependencies; running the file as a script starts the test runner.
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
"""Tests for training_util."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training_util


class GlobalStepTest(test.TestCase):
  # [CLASS DOC]
  # Tests for creating, fetching and validating the global step variable of a graph through
  # get_global_step, create_global_step and get_or_create_global_step. Each test builds its own
  # fresh graph so no state is shared between tests.

  def _assert_global_step(self, global_step, expected_dtype=dtypes.int64):
    # [FUNCTION DOC]
    # HELPER: shared assertion used by the global step tests in this class.
    # Checks that the given variable is named after the global step collection key with output
    # index zero, has the expected base dtype (default 64-bit integer) and is a scalar.
    # Inputs: global_step (the variable under inspection), expected_dtype (optional dtype).
    # Returns nothing; fails the test through assertions on mismatch.
    self.assertEqual('%s:0' % ops.GraphKeys.GLOBAL_STEP, global_step.name)
    self.assertEqual(expected_dtype, global_step.dtype.base_dtype)
    self.assertEqual([], global_step.get_shape().as_list())

  def test_invalid_dtype(self):
    # [FUNCTION DOC]
    # Scenario: inside a new graph, no global step exists yet (lookup gives nothing), then a
    # non-trainable float32 variable is registered in the global step collection.
    # Expected: get_global_step raises a type error saying the variable does not have integer
    # type, both for the default graph and when the graph is passed explicitly.
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variable_v1.VariableV1(
          0.0,
          trainable=False,
          dtype=dtypes.float32,
          name=ops.GraphKeys.GLOBAL_STEP,
          collections=[ops.GraphKeys.GLOBAL_STEP])
      self.assertRaisesRegex(TypeError, 'does not have integer type',
                             training_util.get_global_step)
    self.assertRaisesRegex(TypeError, 'does not have integer type',
                           training_util.get_global_step, g)

  def test_invalid_shape(self):
    # [FUNCTION DOC]
    # Scenario: in a new graph, register an int32 variable of shape one (a vector, not a
    # scalar) in the global step collection.
    # Expected: get_global_step raises a type error mentioning "not scalar", for both the
    # default graph and an explicitly passed graph.
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variable_v1.VariableV1([0],
                             trainable=False,
                             dtype=dtypes.int32,
                             name=ops.GraphKeys.GLOBAL_STEP,
                             collections=[ops.GraphKeys.GLOBAL_STEP])
      self.assertRaisesRegex(TypeError, 'not scalar',
                             training_util.get_global_step)
    self.assertRaisesRegex(TypeError, 'not scalar',
                           training_util.get_global_step, g)

  def test_create_global_step(self):
    # [FUNCTION DOC]
    # Scenario: with no global step in the outer default graph, create one in a new graph.
    # Expected: the created variable has the standard name, int64 dtype and scalar shape; a
    # second creation (default or explicit graph) raises a value error containing "already
    # exists"; creating one in a brand new separate graph succeeds.
    self.assertIsNone(training_util.get_global_step())
    with ops.Graph().as_default() as g:
      global_step = training_util.create_global_step()
      self._assert_global_step(global_step)
      self.assertRaisesRegex(ValueError, 'already exists',
                             training_util.create_global_step)
      self.assertRaisesRegex(ValueError, 'already exists',
                             training_util.create_global_step, g)
      self._assert_global_step(training_util.create_global_step(ops.Graph()))

  def test_get_global_step(self):
    # [FUNCTION DOC]
    # Scenario: in a new graph, first confirm no global step exists, then register a valid
    # int32 scalar variable in the global step collection.
    # Expected: get_global_step returns it (name, int32 dtype, scalar shape), both inside the
    # graph context and later when the graph is passed explicitly.
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variable_v1.VariableV1(
          0,
          trainable=False,
          dtype=dtypes.int32,
          name=ops.GraphKeys.GLOBAL_STEP,
          collections=[ops.GraphKeys.GLOBAL_STEP])
      self._assert_global_step(
          training_util.get_global_step(), expected_dtype=dtypes.int32)
    self._assert_global_step(
        training_util.get_global_step(g), expected_dtype=dtypes.int32)

  def test_get_or_create_global_step(self):
    # [FUNCTION DOC]
    # Scenario: in a new graph with no global step, call get_or_create_global_step.
    # Expected: a valid default (int64 scalar) global step is created, and calling again with
    # the graph passed explicitly returns a variable that passes the same checks.
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      self._assert_global_step(training_util.get_or_create_global_step())
      self._assert_global_step(training_util.get_or_create_global_step(g))


class GlobalStepReadTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the private cached global-step read tensor and the private increment helper in
  # training_util. Each test works in its own fresh graph.

  def test_global_step_read_is_none_if_there_is_no_global_step(self):
    # [FUNCTION DOC]
    # Scenario: in a new graph, request the cached global step read before any global step
    # exists, then create the global step and request it again.
    # Expected: first result is nothing/null; after creation the result is not null.
    with ops.Graph().as_default():
      self.assertIsNone(training_util._get_or_create_global_step_read())
      training_util.create_global_step()
      self.assertIsNotNone(training_util._get_or_create_global_step_read())

  def test_reads_from_cache(self):
    # [FUNCTION DOC]
    # Scenario: create a global step and ask twice for the cached read tensor.
    # Expected: both calls return the same (equal) tensor, showing the read is cached.
    with ops.Graph().as_default():
      training_util.create_global_step()
      first = training_util._get_or_create_global_step_read()
      second = training_util._get_or_create_global_step_read()
      self.assertEqual(first, second)

  def test_reads_before_increments(self):
    # [FUNCTION DOC]
    # Scenario: create a global step, get its cached read tensor and build two increment ops
    # (by one and by three). Run them inside a monitored training session, each together with
    # the read tensor, then read once more alone.
    # Expected: the read observes the value before the increment in the same run: 0, then 1,
    # and finally 4 once both increments have been applied.
    with ops.Graph().as_default():
      training_util.create_global_step()
      read_tensor = training_util._get_or_create_global_step_read()
      inc_op = training_util._increment_global_step(1)
      inc_three_op = training_util._increment_global_step(3)
      with monitored_session.MonitoredTrainingSession() as sess:
        read_value, _ = sess.run([read_tensor, inc_op])
        self.assertEqual(0, read_value)
        read_value, _ = sess.run([read_tensor, inc_three_op])
        self.assertEqual(1, read_value)
        read_value = sess.run(read_tensor)
        self.assertEqual(4, read_value)


if __name__ == '__main__':
  test.main()
