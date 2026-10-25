# [MODULE DOC]
# Tests for the v2 summary ops in summary_ops_v2: writing scalars and raw summary protos, default
# step handling, conditional recording, file writer lifecycle (init, flush, close, naming, sharing),
# saved-model round trips of writers, the no-op writer, and run metadata, trace and graph summaries.
# Includes small helpers that read events back from a log directory. No import-time side effects.

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
"""Tests for V2 summary ops from summary_ops_v2."""

import os
import unittest

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.module import module
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import loader as saved_model_loader
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import tag_constants


class SummaryOpsCoreTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the core v2 summary API: the write and raw-protobuf write functions, default step
  # handling, conditional recording, summary scopes, and the registry of v2 summary ops. Each test
  # uses a temporary directory as event log directory and reads events back from it.

  def testWrite(self):
    # [FUNCTION DOC]
    # Eagerly writes a scalar value with an explicit step through a file writer set as default,
    # asserts the write reports success, then reads back two events (file version plus summary) and
    # checks step, tag and value.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        output = summary_ops.write('tag', 42, step=12)
        self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    value = events[1].summary.value[0]
    self.assertEqual('tag', value.tag)
    self.assertEqual(42, to_numpy(value))

  def testWrite_fromFunction(self):
    # [FUNCTION DOC]
    # Same as the basic write test, but the write happens inside a compiled function that enters the
    # writer's default context. Asserts the returned flag is true and the logged step, tag and value.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: traced function body for the enclosing test.
        # Enters the default context of the enclosing writer and returns the result of writing a scalar
        # with tag and step 12.
        with writer.as_default():
          return summary_ops.write('tag', 42, step=12)
      output = f()
      self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    value = events[1].summary.value[0]
    self.assertEqual('tag', value.tag)
    self.assertEqual(42, to_numpy(value))

  def testWrite_metadata(self):
    # [FUNCTION DOC]
    # Writes three summaries with plugin metadata given as a message object, serialized bytes and a
    # string tensor. Asserts four events were logged and that each summary event carries metadata
    # equal to the original message.
    logdir = self.get_temp_dir()
    metadata = summary_pb2.SummaryMetadata()
    metadata.plugin_data.plugin_name = 'foo'
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.write('obj', 0, 0, metadata=metadata)
        summary_ops.write('bytes', 0, 0, metadata=metadata.SerializeToString())
        m = constant_op.constant(metadata.SerializeToString())
        summary_ops.write('string_tensor', 0, 0, metadata=m)
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(metadata, events[1].summary.value[0].metadata)
    self.assertEqual(metadata, events[2].summary.value[0].metadata)
    self.assertEqual(metadata, events[3].summary.value[0].metadata)

  def testWrite_name(self):
    # [FUNCTION DOC]
    # Inside a compiled function, writes with an explicit op name and asserts the returned op's name
    # starts with that name. The assertion runs at trace time. Calls the function once.
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Performs a named write with no default writer and asserts the resulting op name begins with the
      # given name; returns nothing.
      output = summary_ops.write('tag', 42, step=12, name='anonymous')
      self.assertTrue(output.name.startswith('anonymous'))
    f()

  def testWrite_ndarray(self):
    # [FUNCTION DOC]
    # Writes a 2x2 nested list as the value and asserts the logged tensor equals the same array.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.write('tag', [[1, 2], [3, 4]], step=12)
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual([[1, 2], [3, 4]], to_numpy(value))

  def testWrite_tensor(self):
    # [FUNCTION DOC]
    # Writes an eager tensor value and asserts the stored tensor equals the tensor's numpy value.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      t = constant_op.constant([[1, 2], [3, 4]])
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.write('tag', t, step=12)
      expected = t.numpy()
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual(expected, to_numpy(value))

  def testWrite_tensor_fromFunction(self):
    # [FUNCTION DOC]
    # Passes a tensor into a compiled function that writes it under the writer's default context;
    # asserts the logged tensor equals the tensor's numpy value.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f(t):
        # [FUNCTION DOC]
        # HELPER: traced function body for the enclosing test.
        # Takes a tensor argument t, enters the writer's default context and writes it with step 12.
        # Returns nothing.
        with writer.as_default():
          summary_ops.write('tag', t, step=12)
      t = constant_op.constant([[1, 2], [3, 4]])
      f(t)
      expected = t.numpy()
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual(expected, to_numpy(value))

  def testWrite_stringTensor(self):
    # [FUNCTION DOC]
    # Writes a list of two byte strings and asserts the logged tensor contains the same strings.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.write('tag', [b'foo', b'bar'], step=12)
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual([b'foo', b'bar'], to_numpy(value))

  @test_util.run_gpu_only
  def testWrite_gpuDeviceContext(self):
    # [FUNCTION DOC]
    # GPU-only test. Writes a float value and an int64 step created under a GPU device scope and
    # forces execution. Asserts two events, the step, the value and empty metadata.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with ops.device('/GPU:0'):
          value = constant_op.constant(42.0)
          step = constant_op.constant(12, dtype=dtypes.int64)
          summary_ops.write('tag', value, step=step).numpy()
    empty_metadata = summary_pb2.SummaryMetadata()
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    self.assertEqual(42, to_numpy(events[1].summary.value[0]))
    self.assertEqual(empty_metadata, events[1].summary.value[0].metadata)

  @test_util.also_run_as_tf_function
  def testWrite_noDefaultWriter(self):
    # [FUNCTION DOC]
    # Runs both eagerly and as a compiled function. With no default writer, a write must return a
    # false result. Uses an array-equality assertion because it also works inside a compiled function.
    # Use assertAllEqual instead of assertFalse since it works in a defun.
    self.assertAllEqual(False, summary_ops.write('tag', 42, step=0))

  @test_util.also_run_as_tf_function
  def testWrite_noStep_okayIfAlsoNoDefaultWriter(self):
    # [FUNCTION DOC]
    # Runs eagerly and compiled. With no default writer and no step set, a write must not complain
    # about the missing step and must return false.
    # Use assertAllEqual instead of assertFalse since it works in a defun.
    self.assertAllEqual(False, summary_ops.write('tag', 42))

  def testWrite_noStep(self):
    # [FUNCTION DOC]
    # With a default writer and no step set or given, a write must raise a value error whose message
    # mentions that no step is set.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with self.assertRaisesRegex(ValueError, 'No step set'):
          summary_ops.write('tag', 42)

  def testWrite_noStep_okayIfNotRecordingSummaries(self):
    # [FUNCTION DOC]
    # With a default writer but recording disabled, a write without a step must not raise and must
    # return false.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with summary_ops.record_if(False):
          self.assertFalse(summary_ops.write('tag', 42))

  def testWrite_usingDefaultStep(self):
    # [FUNCTION DOC]
    # Sets the global default step to constants and then to a variable, writing after each change.
    # Asserts events carry steps 1, 2, 10 and 11 (variable incremented). Side effect: always resets
    # the global default step to nothing in a finally block.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        with summary_ops.create_file_writer_v2(logdir).as_default():
          summary_ops.set_step(1)
          summary_ops.write('tag', 1.0)
          summary_ops.set_step(2)
          summary_ops.write('tag', 1.0)
          mystep = variables.Variable(10, dtype=dtypes.int64)
          summary_ops.set_step(mystep)
          summary_ops.write('tag', 1.0)
          mystep.assign_add(1)
          summary_ops.write('tag', 1.0)
      events = events_from_logdir(logdir)
      self.assertEqual(5, len(events))
      self.assertEqual(1, events[1].step)
      self.assertEqual(2, events[2].step)
      self.assertEqual(10, events[3].step)
      self.assertEqual(11, events[4].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepConstant_fromFunction(self):
    # [FUNCTION DOC]
    # A compiled function writes without an explicit step while the global step is set to 1 and then
    # 2. Asserts both events carry step 1 because a constant step is captured at first trace. Resets
    # the global step at the end.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        @def_function.function
        def f():
          # [FUNCTION DOC]
          # HELPER: traced function body for the enclosing test.
          # Enters the writer's default context and writes a scalar without giving a step, relying on the
          # global default step.
          with writer.as_default():
            summary_ops.write('tag', 1.0)
        summary_ops.set_step(1)
        f()
        summary_ops.set_step(2)
        f()
      events = events_from_logdir(logdir)
      self.assertEqual(3, len(events))
      self.assertEqual(1, events[1].step)
      # The step value will still be 1 because the value was captured at the
      # time the function was first traced.
      self.assertEqual(1, events[2].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromFunction(self):
    # [FUNCTION DOC]
    # Like the constant-step test but the default step is a variable, which is read at every call.
    # Asserts logged steps 0, 1 and 10 after assigning to the variable between calls. Resets the
    # global step at the end.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        @def_function.function
        def f():
          # [FUNCTION DOC]
          # HELPER: traced function body for the enclosing test.
          # Enters the writer's default context and writes a scalar without an explicit step.
          with writer.as_default():
            summary_ops.write('tag', 1.0)
        mystep = variables.Variable(0, dtype=dtypes.int64)
        summary_ops.set_step(mystep)
        f()
        mystep.assign_add(1)
        f()
        mystep.assign(10)
        f()
      events = events_from_logdir(logdir)
      self.assertEqual(4, len(events))
      self.assertEqual(0, events[1].step)
      self.assertEqual(1, events[2].step)
      self.assertEqual(10, events[3].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepConstant_fromLegacyGraph(self):
    # [FUNCTION DOC]
    # In legacy graph mode, builds a write op while the global step is 1, changes the step to 2, runs
    # the op twice in a session and flushes. Asserts both events carry step 1 because the value was
    # captured at graph construction. Resets the global step at the end.
    logdir = self.get_temp_dir()
    try:
      with context.graph_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        summary_ops.set_step(1)
        with writer.as_default():
          write_op = summary_ops.write('tag', 1.0)
        summary_ops.set_step(2)
        with self.cached_session() as sess:
          sess.run(writer.init())
          sess.run(write_op)
          sess.run(write_op)
          sess.run(writer.flush())
      events = events_from_logdir(logdir)
      self.assertEqual(3, len(events))
      self.assertEqual(1, events[1].step)
      # The step value will still be 1 because the value was captured at the
      # time the graph was constructed.
      self.assertEqual(1, events[2].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromLegacyGraph(self):
    # [FUNCTION DOC]
    # In legacy graph mode with a variable as global step, runs a write op between increment and
    # assignment ops in a session. Asserts steps 0, 1 and 10 were logged. Resets the global step.
    logdir = self.get_temp_dir()
    try:
      with context.graph_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        mystep = variables.Variable(0, dtype=dtypes.int64)
        summary_ops.set_step(mystep)
        with writer.as_default():
          write_op = summary_ops.write('tag', 1.0)
        first_assign_op = mystep.assign_add(1)
        second_assign_op = mystep.assign(10)
        with self.cached_session() as sess:
          sess.run(writer.init())
          sess.run(mystep.initializer)
          sess.run(write_op)
          sess.run(first_assign_op)
          sess.run(write_op)
          sess.run(second_assign_op)
          sess.run(write_op)
          sess.run(writer.flush())
      events = events_from_logdir(logdir)
      self.assertEqual(4, len(events))
      self.assertEqual(0, events[1].step)
      self.assertEqual(1, events[2].step)
      self.assertEqual(10, events[3].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStep_fromAsDefault(self):
    # [FUNCTION DOC]
    # Nests several default-writer contexts, some giving their own step, and sets a step inside one.
    # Asserts the logged steps follow the innermost scoped step, restoring on exit: 1, 1, 2, 1, 3.
    # Resets the global step at the end.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        with writer.as_default(step=1):
          summary_ops.write('tag', 1.0)
          with writer.as_default():
            summary_ops.write('tag', 1.0)
            with writer.as_default(step=2):
              summary_ops.write('tag', 1.0)
            summary_ops.write('tag', 1.0)
            summary_ops.set_step(3)
          summary_ops.write('tag', 1.0)
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 1, 2, 1, 3], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromAsDefault(self):
    # [FUNCTION DOC]
    # Same nesting scenario as the previous test but the outer step is a variable that is assigned
    # during the test. Asserts logged steps 1, 2, 3, 2, 4. Resets the global step at the end.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        mystep = variables.Variable(1, dtype=dtypes.int64)
        with writer.as_default(step=mystep):
          summary_ops.write('tag', 1.0)
          with writer.as_default():
            mystep.assign(2)
            summary_ops.write('tag', 1.0)
            with writer.as_default(step=3):
              summary_ops.write('tag', 1.0)
            summary_ops.write('tag', 1.0)
            mystep.assign(4)
          summary_ops.write('tag', 1.0)
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 2, 3, 2, 4], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStep_fromSetAsDefault(self):
    # [FUNCTION DOC]
    # Sets the writer as default permanently with a step variable, then with a constant step. Asserts
    # the logged steps are 1, 2, 3. Resets the global step at the end.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        mystep = variables.Variable(1, dtype=dtypes.int64)
        writer.set_as_default(step=mystep)
        summary_ops.write('tag', 1.0)
        mystep.assign(2)
        summary_ops.write('tag', 1.0)
        writer.set_as_default(step=3)
        summary_ops.write('tag', 1.0)
        writer.flush()
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 2, 3], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromSetAsDefault(self):
    # [FUNCTION DOC]
    # Sets the writer as default with steps 1, 2, then with no step argument. Asserts the logged steps
    # are 1, 2, 2 (the previous step is kept). Resets the global step at the end.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer_v2(logdir)
        writer.set_as_default(step=1)
        summary_ops.write('tag', 1.0)
        writer.set_as_default(step=2)
        summary_ops.write('tag', 1.0)
        writer.set_as_default()
        summary_ops.write('tag', 1.0)
        writer.flush()
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 2, 2], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_recordIf_constant(self):
    # [FUNCTION DOC]
    # Checks constant recording conditions: default records, enabled records, disabled does not
    # (returns false). Asserts three events and that the disabled write is absent.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        self.assertTrue(summary_ops.write('default', 1, step=0))
        with summary_ops.record_if(True):
          self.assertTrue(summary_ops.write('set_on', 1, step=0))
        with summary_ops.record_if(False):
          self.assertFalse(summary_ops.write('set_off', 1, step=0))
    events = events_from_logdir(logdir)
    self.assertEqual(3, len(events))
    self.assertEqual('default', events[1].summary.value[0].tag)
    self.assertEqual('set_on', events[2].summary.value[0].tag)

  def testWrite_recordIf_constant_fromFunction(self):
    # [FUNCTION DOC]
    # Same as the constant recording test but inside a compiled function, with array-equality
    # assertions that work during tracing. Asserts only the default and enabled writes were logged.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: traced function body for the enclosing test.
        # Writes three summaries under the default, enabled and disabled recording conditions and checks
        # the returned flags.
        with writer.as_default():
          # Use assertAllEqual instead of assertTrue since it works in a defun.
          self.assertAllEqual(summary_ops.write('default', 1, step=0), True)
          with summary_ops.record_if(True):
            self.assertAllEqual(summary_ops.write('set_on', 1, step=0), True)
          with summary_ops.record_if(False):
            self.assertAllEqual(summary_ops.write('set_off', 1, step=0), False)
      f()
    events = events_from_logdir(logdir)
    self.assertEqual(3, len(events))
    self.assertEqual('default', events[1].summary.value[0].tag)
    self.assertEqual('set_on', events[2].summary.value[0].tag)

  def testWrite_recordIf_callable(self):
    # [FUNCTION DOC]
    # Uses a Python callable as recording condition that increments a step variable and returns true
    # for even values. Asserts alternate writes return true and false and that steps 0, 2, 4 are
    # logged.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      step = variables.Variable(-1, dtype=dtypes.int64)
      def record_fn():
        # [FUNCTION DOC]
        # HELPER: recording-condition callback for the enclosing test.
        # Increments the step variable by one and returns whether its new value is even.
        step.assign_add(1)
        return int(step % 2) == 0
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with summary_ops.record_if(record_fn):
          self.assertTrue(summary_ops.write('tag', 1, step=step))
          self.assertFalse(summary_ops.write('tag', 1, step=step))
          self.assertTrue(summary_ops.write('tag', 1, step=step))
          self.assertFalse(summary_ops.write('tag', 1, step=step))
          self.assertTrue(summary_ops.write('tag', 1, step=step))
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(0, events[1].step)
    self.assertEqual(2, events[2].step)
    self.assertEqual(4, events[3].step)

  def testWrite_recordIf_callable_fromFunction(self):
    # [FUNCTION DOC]
    # Uses a compiled function as the recording condition and another compiled function that performs
    # three writes. Asserts the returned flags alternate across two calls and that steps 0, 2, 4 were
    # logged.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      step = variables.Variable(-1, dtype=dtypes.int64)
      @def_function.function
      def record_fn():
        # [FUNCTION DOC]
        # HELPER: recording-condition callback for the enclosing test.
        # Compiled function that increments the step variable and returns a boolean tensor telling
        # whether the new value is even.
        step.assign_add(1)
        return math_ops.equal(step % 2, 0)
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: traced function body for the enclosing test.
        # Under the writer and the callable recording condition, performs three writes and returns their
        # result flags as a list.
        with writer.as_default():
          with summary_ops.record_if(record_fn):
            return [
                summary_ops.write('tag', 1, step=step),
                summary_ops.write('tag', 1, step=step),
                summary_ops.write('tag', 1, step=step)]
      self.assertAllEqual(f(), [True, False, True])
      self.assertAllEqual(f(), [False, True, False])
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(0, events[1].step)
    self.assertEqual(2, events[2].step)
    self.assertEqual(4, events[3].step)

  def testWrite_recordIf_tensorInput_fromFunction(self):
    # [FUNCTION DOC]
    # A compiled function with a fixed scalar int64 signature uses a tensor condition (step is even)
    # for recording. Asserts alternate calls return true and false and steps 0, 2, 4 are logged.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[], dtype=dtypes.int64)])
      def f(step):
        # [FUNCTION DOC]
        # HELPER: traced function body for the enclosing test.
        # Takes the step as an argument, records only when it is even and returns the write flag.
        with writer.as_default():
          with summary_ops.record_if(math_ops.equal(step % 2, 0)):
            return summary_ops.write('tag', 1, step=step)
      self.assertTrue(f(0))
      self.assertFalse(f(1))
      self.assertTrue(f(2))
      self.assertFalse(f(3))
      self.assertTrue(f(4))
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(0, events[1].step)
    self.assertEqual(2, events[2].step)
    self.assertEqual(4, events[3].step)

  def testWriteRawPb(self):
    # [FUNCTION DOC]
    # Writes a serialized summary protobuf through the raw-protobuf write function at step 12 and
    # asserts success, two events and that the logged summary equals the original message.
    logdir = self.get_temp_dir()
    pb = summary_pb2.Summary()
    pb.value.add().simple_value = 42.0
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        output = summary_ops.write_raw_pb(pb.SerializeToString(), step=12)
        self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    self.assertProtoEquals(pb, events[1].summary)

  def testWriteRawPb_fromFunction(self):
    # [FUNCTION DOC]
    # Same as the raw protobuf test but from within a compiled function.
    logdir = self.get_temp_dir()
    pb = summary_pb2.Summary()
    pb.value.add().simple_value = 42.0
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: traced function body for the enclosing test.
        # Enters the writer's default context and returns the result of a raw protobuf write at step 12.
        with writer.as_default():
          return summary_ops.write_raw_pb(pb.SerializeToString(), step=12)
      output = f()
      self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    self.assertProtoEquals(pb, events[1].summary)

  def testWriteRawPb_multipleValues(self):
    # [FUNCTION DOC]
    # Writes a list of three serialized summaries in one call. Asserts a single event whose summary
    # contains all six values merged in order.
    logdir = self.get_temp_dir()
    pb1 = summary_pb2.Summary()
    pb1.value.add().simple_value = 1.0
    pb1.value.add().simple_value = 2.0
    pb2 = summary_pb2.Summary()
    pb2.value.add().simple_value = 3.0
    pb3 = summary_pb2.Summary()
    pb3.value.add().simple_value = 4.0
    pb3.value.add().simple_value = 5.0
    pb3.value.add().simple_value = 6.0
    pbs = [pb.SerializeToString() for pb in (pb1, pb2, pb3)]
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        output = summary_ops.write_raw_pb(pbs, step=12)
        self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    expected_pb = summary_pb2.Summary()
    for i in range(6):
      expected_pb.value.add().simple_value = i + 1.0
    self.assertProtoEquals(expected_pb, events[1].summary)

  def testWriteRawPb_invalidValue(self):
    # [FUNCTION DOC]
    # Writing a string that is not a valid serialized summary must raise a data-loss error with a
    # message about a bad summary proto string.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with self.assertRaisesRegex(
            errors.DataLossError,
            'Bad tf.compat.v1.Summary binary proto tensor string'):
          summary_ops.write_raw_pb('notaproto', step=12)

  @test_util.also_run_as_tf_function
  def testGetSetStep(self):
    # [FUNCTION DOC]
    # Runs eagerly and compiled. Asserts the default step starts out empty, then reads back an int
    # and a tensor step after setting them. Resets the global step to nothing at the end.
    try:
      self.assertIsNone(summary_ops.get_step())
      summary_ops.set_step(1)
      # Use assertAllEqual instead of assertEqual since it works in a defun.
      self.assertAllEqual(1, summary_ops.get_step())
      summary_ops.set_step(constant_op.constant(2))
      self.assertAllEqual(2, summary_ops.get_step())
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testGetSetStep_variable(self):
    # [FUNCTION DOC]
    # Sets a variable as the default step and asserts reads track assignments, that the global state
    # keeps the variable alive after the local name is deleted, and that incrementing through the
    # getter works. Resets the global step at the end.
    with context.eager_mode():
      try:
        mystep = variables.Variable(0)
        summary_ops.set_step(mystep)
        self.assertAllEqual(0, summary_ops.get_step().read_value())
        mystep.assign_add(1)
        self.assertAllEqual(1, summary_ops.get_step().read_value())
        # Check that set_step() properly maintains reference to variable.
        del mystep
        self.assertAllEqual(1, summary_ops.get_step().read_value())
        summary_ops.get_step().assign_add(1)
        self.assertAllEqual(2, summary_ops.get_step().read_value())
      finally:
        # Reset to default state for other tests.
        summary_ops.set_step(None)

  def testGetSetStep_variable_fromFunction(self):
    # [FUNCTION DOC]
    # Sets and increments the default step variable from compiled functions and asserts values are
    # 0, 1, 2 and 3, including after the local reference is deleted. Resets the global step.
    with context.eager_mode():
      try:
        @def_function.function
        def set_step(step):
          # [FUNCTION DOC]
          # HELPER: compiled function used by the enclosing test.
          # Sets the default step to the given argument and returns the current default step.
          summary_ops.set_step(step)
          return summary_ops.get_step()
        @def_function.function
        def get_and_increment():
          # [FUNCTION DOC]
          # HELPER: compiled function used by the enclosing test.
          # Increments the default step variable by one and returns the step.
          summary_ops.get_step().assign_add(1)
          return summary_ops.get_step()
        mystep = variables.Variable(0)
        self.assertAllEqual(0, set_step(mystep))
        self.assertAllEqual(0, summary_ops.get_step().read_value())
        self.assertAllEqual(1, get_and_increment())
        self.assertAllEqual(2, get_and_increment())
        # Check that set_step() properly maintains reference to variable.
        del mystep
        self.assertAllEqual(3, get_and_increment())
      finally:
        # Reset to default state for other tests.
        summary_ops.set_step(None)

  @test_util.also_run_as_tf_function
  def testSummaryScope(self):
    # [FUNCTION DOC]
    # Runs eagerly and compiled. Asserts the scope helper yields a tag and scope string for plain
    # names, nested names, names with slashes, and that a null outer name scope resets nesting.
    with summary_ops.summary_scope('foo') as (tag, scope):
      self.assertEqual('foo', tag)
      self.assertEqual('foo/', scope)
      with summary_ops.summary_scope('bar') as (tag, scope):
        self.assertEqual('foo/bar', tag)
        self.assertEqual('foo/bar/', scope)
      with summary_ops.summary_scope('with/slash') as (tag, scope):
        self.assertEqual('foo/with/slash', tag)
        self.assertEqual('foo/with/slash/', scope)
      with ops.name_scope(None, skip_on_eager=False):
        with summary_ops.summary_scope('unnested') as (tag, scope):
          self.assertEqual('unnested', tag)
          self.assertEqual('unnested/', scope)

  @test_util.also_run_as_tf_function
  def testSummaryScope_defaultName(self):
    # [FUNCTION DOC]
    # Runs eagerly and compiled. With no name, the scope uses the default name or the given backup
    # name; asserts both the tag and scope string.
    with summary_ops.summary_scope(None) as (tag, scope):
      self.assertEqual('summary', tag)
      self.assertEqual('summary/', scope)
    with summary_ops.summary_scope(None, 'backup') as (tag, scope):
      self.assertEqual('backup', tag)
      self.assertEqual('backup/', scope)

  @test_util.also_run_as_tf_function
  def testSummaryScope_handlesCharactersIllegalForScope(self):
    # [FUNCTION DOC]
    # Runs eagerly and compiled. Illegal scope characters are dropped from the scope name but kept in
    # the tag; if nothing legal remains the backup name is used for the scope. Asserts both.
    with summary_ops.summary_scope('f?o?o') as (tag, scope):
      self.assertEqual('f?o?o', tag)
      self.assertEqual('foo/', scope)
    # If all characters aren't legal for a scope name, use default name.
    with summary_ops.summary_scope('???', 'backup') as (tag, scope):
      self.assertEqual('???', tag)
      self.assertEqual('backup/', scope)

  @test_util.also_run_as_tf_function
  def testSummaryScope_nameNotUniquifiedForTag(self):
    # [FUNCTION DOC]
    # Runs eagerly and compiled. Creates ops with colliding names first and asserts the tag from the
    # summary scope is not made unique, including for names containing slashes.
    constant_op.constant(0, name='foo')
    with summary_ops.summary_scope('foo') as (tag, _):
      self.assertEqual('foo', tag)
    with summary_ops.summary_scope('foo') as (tag, _):
      self.assertEqual('foo', tag)
    with ops.name_scope('with', skip_on_eager=False):
      constant_op.constant(0, name='slash')
    with summary_ops.summary_scope('with/slash') as (tag, _):
      self.assertEqual('with/slash', tag)

  def testAllV2SummaryOps(self):
    # [FUNCTION DOC]
    # In graph mode, defines a set of v2 and legacy summary ops with no writer, with recording on and
    # with recording off. Asserts the global registry returns exactly the ops created while a default
    # writer was present, regardless of recording state.
    logdir = self.get_temp_dir()
    def define_ops():
      # [FUNCTION DOC]
      # HELPER: builds one op of each summary kind for the enclosing test.
      # Creates write, raw protobuf, generic, scalar, histogram, image and audio summary ops and
      # returns them as a list.
      result = []
      # TF 2.0 summary ops
      result.append(summary_ops.write('write', 1, step=0))
      result.append(summary_ops.write_raw_pb(b'', step=0, name='raw_pb'))
      # TF 1.x tf.contrib.summary ops
      result.append(summary_ops.generic('tensor', 1, step=1))
      result.append(summary_ops.scalar('scalar', 2.0, step=1))
      result.append(summary_ops.histogram('histogram', [1.0], step=1))
      result.append(summary_ops.image('image', [[[[1.0]]]], step=1))
      result.append(summary_ops.audio('audio', [[1.0]], 1.0, 1, step=1))
      return result
    with context.graph_mode():
      ops_without_writer = define_ops()
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with summary_ops.record_if(True):
          ops_recording_on = define_ops()
        with summary_ops.record_if(False):
          ops_recording_off = define_ops()
      # We should be collecting all ops defined with a default writer present,
      # regardless of whether recording was set on or off, but not those defined
      # without a writer at all.
      del ops_without_writer
      expected_ops = ops_recording_on + ops_recording_off
      self.assertCountEqual(expected_ops, summary_ops.all_v2_summary_ops())

  def testShouldRecordSummaries_defaultState(self):
    # [FUNCTION DOC]
    # Asserts the should-record query is false without a default writer (even with recording
    # enabled), and true only while a writer is the default.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      self.assertAllEqual(False, summary_ops.should_record_summaries())
      w = summary_ops.create_file_writer_v2(logdir)
      self.assertAllEqual(False, summary_ops.should_record_summaries())
      with w.as_default():
        # Should be enabled only when default writer is registered.
        self.assertAllEqual(True, summary_ops.should_record_summaries())
      self.assertAllEqual(False, summary_ops.should_record_summaries())
      with summary_ops.record_if(True):
        # Should be disabled when no default writer, even with record_if(True).
        self.assertAllEqual(False, summary_ops.should_record_summaries())

  def testShouldRecordSummaries_constants(self):
    # [FUNCTION DOC]
    # Asserts the should-record query follows nested constant recording conditions.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with summary_ops.record_if(True):
          self.assertAllEqual(True, summary_ops.should_record_summaries())
        with summary_ops.record_if(False):
          self.assertAllEqual(False, summary_ops.should_record_summaries())
          with summary_ops.record_if(True):
            self.assertAllEqual(True, summary_ops.should_record_summaries())

  def testShouldRecordSummaries_variable(self):
    # [FUNCTION DOC]
    # Asserts the query follows a boolean variable condition as it is assigned.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        cond = variables.Variable(False)
        with summary_ops.record_if(cond):
          self.assertAllEqual(False, summary_ops.should_record_summaries())
          cond.assign(True)
          self.assertAllEqual(True, summary_ops.should_record_summaries())

  def testShouldRecordSummaries_callable(self):
    # [FUNCTION DOC]
    # Asserts the query re-evaluates a callable condition every time.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        cond_box = [False]
        cond = lambda: cond_box[0]
        with summary_ops.record_if(cond):
          self.assertAllEqual(False, summary_ops.should_record_summaries())
          cond_box[0] = True
          self.assertAllEqual(True, summary_ops.should_record_summaries())

  def testShouldRecordSummaries_fromFunction(self):
    # [FUNCTION DOC]
    # Inside a compiled function with a boolean argument, collects the query result without a writer,
    # with a writer, with recording off and with a tensor condition. Asserts the results for true and
    # false argument values.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[], dtype=dtypes.bool)])
      def f(cond):
        # [FUNCTION DOC]
        # HELPER: traced function body for the enclosing test.
        # Takes a boolean condition and returns a list of four query results under different contexts.
        results = []
        results.append(summary_ops.should_record_summaries())
        with writer.as_default():
          results.append(summary_ops.should_record_summaries())
          with summary_ops.record_if(False):
            results.append(summary_ops.should_record_summaries())
          with summary_ops.record_if(cond):
            results.append(summary_ops.should_record_summaries())
        return results
      self.assertAllEqual([False, True, False, True], f(True))
      self.assertAllEqual([False, True, False, False], f(False))

  def testHasDefaultWriter_checkWriter(self):
    # [FUNCTION DOC]
    # Uses subtests: has-default-writer is true inside a writer's context and false outside.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with self.subTest(name='has_writer'):
        with summary_ops.create_file_writer_v2(logdir).as_default():
          self.assertTrue(summary_ops.has_default_writer())
      with self.subTest(name='no_writer'):
        self.assertFalse(summary_ops.has_default_writer())


class SummaryWriterTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the file writer objects: creation, flushing, closing, queueing, file name collision
  # avoidance, sharing and memory behavior.

  def testCreate_withInitAndClose(self):
    # [FUNCTION DOC]
    # Creates a writer with a large queue. Asserts the file starts with one version event, that a
    # second init has no effect, that a write stays buffered and that closing flushes it.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(
          logdir, max_queue=1000, flush_millis=1000000)
      get_total = lambda: len(events_from_logdir(logdir))
      self.assertEqual(1, get_total())  # file_version Event
      # Calling init() again while writer is open has no effect
      writer.init()
      self.assertEqual(1, get_total())
      with writer.as_default():
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        # Calling .close() should do an implicit flush
        writer.close()
        self.assertEqual(2, get_total())

  def testCreate_fromFunction(self):
    # [FUNCTION DOC]
    # Creates a writer inside a compiled function and keeps it in a function attribute so it stays
    # alive. Asserts exactly one event file is created.
    logdir = self.get_temp_dir()
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Creates a file writer once and stores it on the function object to keep it alive.
      # Returned SummaryWriter must be stored in a non-local variable so it
      # lives throughout the function execution.
      if not hasattr(f, 'writer'):
        f.writer = summary_ops.create_file_writer_v2(logdir)
    with context.eager_mode():
      f()
    event_files = gfile.Glob(os.path.join(logdir, '*'))
    self.assertEqual(1, len(event_files))

  def testCreate_graphTensorArgument_raisesError(self):
    # [FUNCTION DOC]
    # Passing a graph tensor as log directory to the writer constructor in eager mode must raise a
    # value error about an invalid graph tensor argument, and no file must be created.
    logdir = self.get_temp_dir()
    with context.graph_mode():
      logdir_tensor = constant_op.constant(logdir)
    with context.eager_mode():
      with self.assertRaisesRegex(
          ValueError, 'Invalid graph Tensor argument.*logdir'):
        summary_ops.create_file_writer_v2(logdir_tensor)
    self.assertEmpty(gfile.Glob(os.path.join(logdir, '*')))

  def testCreate_fromFunction_graphTensorArgument_raisesError(self):
    # [FUNCTION DOC]
    # Same as the previous test but the constructor is called in a compiled function with a constant
    # tensor directory; expects the value error and no file.
    logdir = self.get_temp_dir()
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Creates a writer using a constant tensor as log directory, which is invalid.
      summary_ops.create_file_writer_v2(constant_op.constant(logdir))
    with context.eager_mode():
      with self.assertRaisesRegex(
          ValueError, 'Invalid graph Tensor argument.*logdir'):
        f()
    self.assertEmpty(gfile.Glob(os.path.join(logdir, '*')))

  def testCreate_fromFunction_unpersistedResource_raisesError(self):
    # [FUNCTION DOC]
    # A compiled function creates a writer and only enters its default context without keeping it.
    # Expects a not-found error about a missing resource, but still one event file created.
    logdir = self.get_temp_dir()
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Creates a writer and enters its default context without storing it.
      with summary_ops.create_file_writer_v2(logdir).as_default():
        pass  # Calling .as_default() is enough to indicate use.
    with context.eager_mode():
      # TODO(nickfelt): change this to a better error
      with self.assertRaisesRegex(
          errors.NotFoundError, 'Resource.*does not exist'):
        f()
    # Even though we didn't use it, an event file will have been created.
    self.assertEqual(1, len(gfile.Glob(os.path.join(logdir, '*'))))

  def testCreate_immediateSetAsDefault_retainsReference(self):
    # [FUNCTION DOC]
    # Creates a writer and immediately sets it as default without keeping a local reference, then
    # flushes. Passes if flush works, showing the global state keeps it alive. Clears the global
    # writer in a finally block.
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        summary_ops.create_file_writer_v2(logdir).set_as_default()
        summary_ops.flush()
    finally:
      # Ensure we clean up no matter how the test executes.
      summary_ops._summary_state.writer = None  # pylint: disable=protected-access

  def testCreate_immediateAsDefault_retainsReference(self):
    # [FUNCTION DOC]
    # Enters the default context of a temporary writer and flushes inside; passes if no error occurs.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.flush()

  def testCreate_avoidsFilenameCollision(self):
    # [FUNCTION DOC]
    # Creates ten writers on one directory eagerly and asserts ten distinct files exist.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      for _ in range(10):
        summary_ops.create_file_writer_v2(logdir)
    event_files = gfile.Glob(os.path.join(logdir, '*'))
    self.assertLen(event_files, 10)

  def testCreate_graphMode_avoidsFilenameCollision(self):
    # [FUNCTION DOC]
    # In graph mode, repeatedly initializes and closes one writer ten times in a session and asserts
    # ten distinct event files exist.
    logdir = self.get_temp_dir()
    with context.graph_mode(), ops.Graph().as_default():
      writer = summary_ops.create_file_writer_v2(logdir)
      with self.cached_session() as sess:
        for _ in range(10):
          sess.run(writer.init())
          sess.run(writer.close())
    event_files = gfile.Glob(os.path.join(logdir, '*'))
    self.assertLen(event_files, 10)

  def testNoSharing(self):
    # [FUNCTION DOC]
    # Two writers on the same directory must not share state. Interleaves writes and asserts each
    # file holds its own version event and its own steps (1 and 2), with nothing more.
    # Two writers with the same logdir should not share state.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer1 = summary_ops.create_file_writer_v2(logdir)
      with writer1.as_default():
        summary_ops.write('tag', 1, step=1)
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(event_files))
      file1 = event_files[0]

      writer2 = summary_ops.create_file_writer_v2(logdir)
      with writer2.as_default():
        summary_ops.write('tag', 1, step=2)
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(2, len(event_files))
      event_files.remove(file1)
      file2 = event_files[0]

      # Extra writes to ensure interleaved usage works.
      with writer1.as_default():
        summary_ops.write('tag', 1, step=1)
      with writer2.as_default():
        summary_ops.write('tag', 1, step=2)

    events = iter(events_from_file(file1))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(1, next(events).step)
    self.assertEqual(1, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))
    events = iter(events_from_file(file2))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(2, next(events).step)
    self.assertEqual(2, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))

  def testNoSharing_fromFunction(self):
    # [FUNCTION DOC]
    # Same non-sharing scenario with writers created lazily inside two compiled functions; asserts
    # each file contains only its own events.
    logdir = self.get_temp_dir()
    @def_function.function
    def f1():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Lazily creates a writer stored on itself and writes a scalar with step 1.
      if not hasattr(f1, 'writer'):
        f1.writer = summary_ops.create_file_writer_v2(logdir)
      with f1.writer.as_default():
        summary_ops.write('tag', 1, step=1)
    @def_function.function
    def f2():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Lazily creates a writer stored on itself and writes a scalar with step 2.
      if not hasattr(f2, 'writer'):
        f2.writer = summary_ops.create_file_writer_v2(logdir)
      with f2.writer.as_default():
        summary_ops.write('tag', 1, step=2)
    with context.eager_mode():
      f1()
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(event_files))
      file1 = event_files[0]

      f2()
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(2, len(event_files))
      event_files.remove(file1)
      file2 = event_files[0]

      # Extra writes to ensure interleaved usage works.
      f1()
      f2()

    events = iter(events_from_file(file1))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(1, next(events).step)
    self.assertEqual(1, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))
    events = iter(events_from_file(file2))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(2, next(events).step)
    self.assertEqual(2, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))

  def testMaxQueue(self):
    # [FUNCTION DOC]
    # With a queue of one, the second write forces a flush. Asserts the event counts 1, 1, then 3.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(
          logdir, max_queue=1, flush_millis=999999).as_default():
        get_total = lambda: len(events_from_logdir(logdir))
        # Note: First tf.compat.v1.Event is always file_version.
        self.assertEqual(1, get_total())
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        # Should flush after second summary since max_queue = 1
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(3, get_total())

  def testWriterFlush(self):
    # [FUNCTION DOC]
    # Asserts that explicit flushing and leaving the default context flush buffered events, with the
    # count of events read from disk changing from 1 to 2, 2, then 3.
    logdir = self.get_temp_dir()
    get_total = lambda: len(events_from_logdir(logdir))
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(
          logdir, max_queue=1000, flush_millis=1000000)
      self.assertEqual(1, get_total())  # file_version Event
      with writer.as_default():
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        writer.flush()
        self.assertEqual(2, get_total())
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(2, get_total())
      # Exiting the "as_default()" should do an implicit flush
      self.assertEqual(3, get_total())

  def testFlushFunction(self):
    # [FUNCTION DOC]
    # Tests the module-level flush function: without arguments it flushes the default writer, and
    # with an explicit writer argument it flushes that writer. Asserts event counts 1, 3, 3, 4.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(
          logdir, max_queue=999999, flush_millis=999999)
      with writer.as_default():
        get_total = lambda: len(events_from_logdir(logdir))
        # Note: First tf.compat.v1.Event is always file_version.
        self.assertEqual(1, get_total())
        summary_ops.write('tag', 1, step=0)
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        summary_ops.flush()
        self.assertEqual(3, get_total())
        # Test "writer" parameter
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(3, get_total())
        summary_ops.flush(writer=writer)
        self.assertEqual(4, get_total())

  # Regression test for b/228097117.
  def testFlushFunction_disallowsInvalidWriterInput(self):
    # [FUNCTION DOC]
    # Regression test: flushing with an invalid writer argument (an empty tuple) must raise a value
    # error saying the argument to flush is invalid.
    with context.eager_mode():
      with self.assertRaisesRegex(ValueError, 'Invalid argument to flush'):
        summary_ops.flush(writer=())

  @test_util.assert_no_new_tensors
  def testNoMemoryLeak_graphMode(self):
    # [FUNCTION DOC]
    # Creates a writer in graph mode and relies on the leak-check decorator to fail if new tensors
    # remain.
    logdir = self.get_temp_dir()
    with context.graph_mode(), ops.Graph().as_default():
      summary_ops.create_file_writer_v2(logdir)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNoMemoryLeak_eagerMode(self):
    # [FUNCTION DOC]
    # Creates a writer and writes once in eager mode; the decorator fails if new Python objects
    # remain afterwards.
    logdir = self.get_temp_dir()
    with summary_ops.create_file_writer_v2(logdir).as_default():
      summary_ops.write('tag', 1, step=0)

  def testClose_preventsLaterUse(self):
    # [FUNCTION DOC]
    # After closing, a second close and a flush are silent no-ops, but init, entering the default
    # context and setting as default must each raise a runtime error mentioning it is already closed.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      writer.close()
      writer.close()  # redundant close() is a no-op
      writer.flush()  # redundant flush() is a no-op
      with self.assertRaisesRegex(RuntimeError, 'already closed'):
        writer.init()
      with self.assertRaisesRegex(RuntimeError, 'already closed'):
        with writer.as_default():
          self.fail('should not get here')
      with self.assertRaisesRegex(RuntimeError, 'already closed'):
        writer.set_as_default()

  def testClose_closesOpenFile(self):
    # [FUNCTION DOC]
    # Skipped if psutil is missing. Asserts the writer's event file appears in the process's open
    # files and disappears after closing the writer.
    try:
      import psutil  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise unittest.SkipTest('test requires psutil')
    proc = psutil.Process()
    get_open_filenames = lambda: set(info[0] for info in proc.open_files())
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(files))
      eventfile = files[0]
      self.assertIn(eventfile, get_open_filenames())
      writer.close()
      self.assertNotIn(eventfile, get_open_filenames())

  def testDereference_closesOpenFile(self):
    # [FUNCTION DOC]
    # Skipped if psutil is missing. Asserts that deleting the last reference to a writer closes its
    # open event file.
    try:
      import psutil  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise unittest.SkipTest('test requires psutil')
    proc = psutil.Process()
    get_open_filenames = lambda: set(info[0] for info in proc.open_files())
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(files))
      eventfile = files[0]
      self.assertIn(eventfile, get_open_filenames())
      del writer
      self.assertNotIn(eventfile, get_open_filenames())


class SummaryWriterSavedModelTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that a file writer stored on a module survives saving and loading a saved model, in eager
  # and graph mode, with the expected number of event files and events.

  def testWriter_savedAsModuleProperty_loadInEagerMode(self):
    # [FUNCTION DOC]
    # Defines a module with a trackable writer and a training function with a scalar signature, saves
    # it, resets the global eager context and loads it twice. Asserts three event files exist (one
    # from the original, one per load) with steps 1,2 and 3,4.
    with context.eager_mode():
      class Model(module.Module):
        # [CLASS DOC]
        # Module holding a trackable file writer and a train function that logs a summary at a given step.
        # Defined locally in the test.

        def __init__(self, model_dir):
          # [FUNCTION DOC]
          # Creates a trackable file writer in the given model directory and stores it as an attribute.
          self._writer = summary_ops.create_file_writer_v2(
              model_dir, experimental_trackable=True)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[], dtype=dtypes.int64)
        ])
        def train(self, step):
          # [FUNCTION DOC]
          # Compiled function with a scalar int64 signature: writes a string summary at the given step and
          # returns a constant zero.
          with self._writer.as_default():
            summary_ops.write('tag', 'foo', step=step)
          return constant_op.constant(0)

      logdir = self.get_temp_dir()
      to_export = Model(logdir)
      pre_save_files = set(events_from_multifile_logdir(logdir))
      export_dir = os.path.join(logdir, 'export')
      saved_model_save.save(
          to_export, export_dir, signatures={'train': to_export.train})

    # Reset context to ensure we don't share any resources with saving code.
    context._reset_context()  # pylint: disable=protected-access
    with context.eager_mode():
      restored = saved_model_load.load(export_dir)
      restored.train(1)
      restored.train(2)
      post_restore_files = set(events_from_multifile_logdir(logdir))
      restored2 = saved_model_load.load(export_dir)
      restored2.train(3)
      restored2.train(4)
      files_to_events = events_from_multifile_logdir(logdir)
      post_restore2_files = set(files_to_events)
      self.assertLen(files_to_events, 3)
      def unwrap_singleton(iterable):
        # [FUNCTION DOC]
        # HELPER: test utility for the enclosing test.
        # Asserts the iterable has exactly one element and returns it.
        self.assertLen(iterable, 1)
        return next(iter(iterable))
      restore_file = unwrap_singleton(post_restore_files - pre_save_files)
      restore2_file = unwrap_singleton(post_restore2_files - post_restore_files)
      restore_events = files_to_events[restore_file]
      restore2_events = files_to_events[restore2_file]
      self.assertLen(restore_events, 3)
      self.assertEqual(1, restore_events[1].step)
      self.assertEqual(2, restore_events[2].step)
      self.assertLen(restore2_events, 3)
      self.assertEqual(3, restore2_events[1].step)
      self.assertEqual(4, restore2_events[2].step)

  def testWriter_savedAsModuleProperty_loadInGraphMode(self):
    # [FUNCTION DOC]
    # Saves the same module in eager mode, then loads it in legacy graph sessions. Within one session
    # all loads share a single writer resource; a new session creates a second file. Asserts three
    # files in total with 5 and 3 events.
    with context.eager_mode():

      class Model(module.Module):
        # [CLASS DOC]
        # Module holding a trackable file writer and a train function that logs a summary at a given step.
        # Defined locally in the test.

        def __init__(self, model_dir):
          # [FUNCTION DOC]
          # Creates a trackable file writer in the given model directory and stores it as an attribute.
          self._writer = summary_ops.create_file_writer_v2(
              model_dir, experimental_trackable=True)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[], dtype=dtypes.int64)
        ])
        def train(self, step):
          # [FUNCTION DOC]
          # Compiled function with a scalar int64 signature: writes a string summary at the given step and
          # returns a constant zero.
          with self._writer.as_default():
            summary_ops.write('tag', 'foo', step=step)
          return constant_op.constant(0)

      logdir = self.get_temp_dir()
      to_export = Model(logdir)
      pre_save_files = set(events_from_multifile_logdir(logdir))
      export_dir = os.path.join(logdir, 'export')
      saved_model_save.save(
          to_export, export_dir, signatures={'train': to_export.train})

    # Reset context to ensure we don't share any resources with saving code.
    context._reset_context()  # pylint: disable=protected-access

    def load_and_run_model(sess, input_values):
      # [FUNCTION DOC]
      # HELPER: runs the saved model in the old style for the enclosing test.
      # Loads the saved model into the session, finds the single input and output tensors of the train
      # signature, and runs the output once per value in input_values. Asserts one input and one output.
      """Load and run the SavedModel signature in the TF 1.x style."""
      model = saved_model_loader.load(sess, [tag_constants.SERVING], export_dir)
      signature = model.signature_def['train']
      inputs = list(signature.inputs.values())
      assert len(inputs) == 1, inputs
      outputs = list(signature.outputs.values())
      assert len(outputs) == 1, outputs
      input_tensor = sess.graph.get_tensor_by_name(inputs[0].name)
      output_tensor = sess.graph.get_tensor_by_name(outputs[0].name)
      for v in input_values:
        sess.run(output_tensor, feed_dict={input_tensor: v})

    with context.graph_mode(), ops.Graph().as_default():
      # Since writer shared_name is fixed, within a single session, all loads of
      # this SavedModel will refer to a single writer resouce, so it will be
      # initialized only once and write to a single file.
      with self.session() as sess:
        load_and_run_model(sess, [1, 2])
        load_and_run_model(sess, [3, 4])
      post_restore_files = set(events_from_multifile_logdir(logdir))
      # New session will recreate the resource and write to a second file.
      with self.session() as sess:
        load_and_run_model(sess, [5, 6])
      files_to_events = events_from_multifile_logdir(logdir)
      post_restore2_files = set(files_to_events)

    self.assertLen(files_to_events, 3)
    def unwrap_singleton(iterable):
      # [FUNCTION DOC]
      # HELPER: test utility for the enclosing test.
      # Asserts the iterable has exactly one element and returns it.
      self.assertLen(iterable, 1)
      return next(iter(iterable))
    restore_file = unwrap_singleton(post_restore_files - pre_save_files)
    restore2_file = unwrap_singleton(post_restore2_files - post_restore_files)
    restore_events = files_to_events[restore_file]
    restore2_events = files_to_events[restore2_file]
    self.assertLen(restore_events, 5)
    self.assertEqual(1, restore_events[1].step)
    self.assertEqual(2, restore_events[2].step)
    self.assertEqual(3, restore_events[3].step)
    self.assertEqual(4, restore_events[4].step)
    self.assertLen(restore2_events, 3)
    self.assertEqual(5, restore2_events[1].step)
    self.assertEqual(6, restore2_events[2].step)


class NoopWriterTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the no-op writer, which accepts all writer calls and writes nothing.

  def testNoopWriter_doesNothing(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the temporary directory is never given to the no-op writer, so the final assertion
    # that the directory has no files is vacuous.
    # Calls init, default context, write, flush and close on the no-op writer. Asserts the write
    # returned false because no active writer was found.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_noop_writer()
      writer.init()
      with writer.as_default():
        result = summary_ops.write('test', 1.0, step=0)
      writer.flush()
      writer.close()
    self.assertFalse(result)  # Should have found no active writer
    files = gfile.Glob(os.path.join(logdir, '*'))
    self.assertLen(files, 0)

  def testNoopWriter_asNestedContext_isTransparent(self):
    # [FUNCTION DOC]
    # Nests the no-op writer inside a real writer's default context. Asserts all three writes inside
    # and around it succeed, so the no-op context is transparent.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      noop_writer = summary_ops.create_noop_writer()
      with writer.as_default():
        result1 = summary_ops.write('first', 1.0, step=0)
        with noop_writer.as_default():
          result2 = summary_ops.write('second', 1.0, step=0)
        result3 = summary_ops.write('third', 1.0, step=0)
    # All ops should have written, including the one inside the no-op writer,
    # since it doesn't actively *disable* writing - it just behaves as if that
    # entire `with` block wasn't there at all.
    self.assertAllEqual([result1, result2, result3], [True, True, True])

  def testNoopWriter_setAsDefault(self):
    # [FUNCTION DOC]
    # Sets a no-op writer as default and asserts a write returns false. Resets the global writer to
    # nothing in a finally block.
    try:
      with context.eager_mode():
        writer = summary_ops.create_noop_writer()
        writer.set_as_default()
        result = summary_ops.write('test', 1.0, step=0)
      self.assertFalse(result)  # Should have found no active writer
    finally:
      # Ensure we clean up no matter how the test executes.
      summary_ops._summary_state.writer = None  # pylint: disable=protected-access


class SummaryOpsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the run metadata, trace and graph summary functions. Provides helpers that write via a
  # temporary file writer and read back the first summary event.

  def tearDown(self):
    # [FUNCTION DOC]
    # Turns tracing off, then calls the base tear down.
    summary_ops.trace_off()
    super().tearDown()

  def exec_summary_op(self, summary_op_fn):
    # [FUNCTION DOC]
    # HELPER: runs the given summary function with a temporary writer in eager mode.
    # Creates a writer, calls the no-argument callback in its default context, closes it, and returns
    # the second event (the first after the version event) from the log directory.
    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer_v2(logdir)
    with writer.as_default():
      summary_op_fn()
    writer.close()
    events = events_from_logdir(logdir)
    return events[1]

  def run_metadata(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: writes run metadata through a temporary writer.
    # Forwards its positional and keyword arguments to the run metadata summary function and returns
    # the first summary event read from the log directory. Requires eager mode.
    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer_v2(logdir)
    with writer.as_default():
      summary_ops.run_metadata(*args, **kwargs)
    writer.close()
    events = events_from_logdir(logdir)
    return events[1]

  def run_metadata_graphs(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: writes run metadata graphs through a temporary writer.
    # Forwards its arguments to the graphs-only run metadata function and returns the first summary
    # event. Requires eager mode.
    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer_v2(logdir)
    with writer.as_default():
      summary_ops.run_metadata_graphs(*args, **kwargs)
    writer.close()
    events = events_from_logdir(logdir)
    return events[1]

  def create_run_metadata(self):
    # [FUNCTION DOC]
    # HELPER: builds a sample run metadata message.
    # Returns a run metadata message with one device step-stats entry for a node named hello and one
    # function graph containing a node named foo.
    step_stats = step_stats_pb2.StepStats(dev_stats=[
        step_stats_pb2.DeviceStepStats(
            device='cpu:0',
            node_stats=[step_stats_pb2.NodeExecStats(node_name='hello')])
    ])
    return config_pb2.RunMetadata(
        function_graphs=[
            config_pb2.RunMetadata.FunctionGraphs(
                pre_optimization_graph=graph_pb2.GraphDef(
                    node=[node_def_pb2.NodeDef(name='foo')]))
        ],
        step_stats=step_stats)

  def run_trace(self, f, step=1):
    # [FUNCTION DOC]
    # HELPER: traces a function and exports it.
    # Parameters: f is a zero-argument callable, step defaults to 1. Starts tracing with graphs and
    # no profiler, runs f under a writer, exports the trace and returns the first summary event.
    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer_v2(logdir)
    summary_ops.trace_on(graph=True, profiler=False)
    with writer.as_default():
      f()
      summary_ops.trace_export(name='foo', step=step)
    writer.close()
    events = events_from_logdir(logdir)
    return events[1]

  @test_util.run_v2_only
  def testRunMetadata_usesNameAsTag(self):
    # [FUNCTION DOC]
    # Asserts the logged summary tag is the name prefixed by the enclosing name scope.
    meta = config_pb2.RunMetadata()

    with ops.name_scope('foo', skip_on_eager=False):
      event = self.run_metadata(name='my_name', data=meta, step=1)
      first_val = event.summary.value[0]

    self.assertEqual('foo/my_name', first_val.tag)

  @test_util.run_v2_only
  def testRunMetadata_summaryMetadata(self):
    # [FUNCTION DOC]
    # Asserts the summary's metadata names the run-metadata plugin with content 1.
    expected_summary_metadata = """
      plugin_data {
        plugin_name: "graph_run_metadata"
        content: "1"
      }
    """
    meta = config_pb2.RunMetadata()
    event = self.run_metadata(name='my_name', data=meta, step=1)
    actual_summary_metadata = event.summary.value[0].metadata
    self.assertProtoEquals(expected_summary_metadata, actual_summary_metadata)

  @test_util.run_v2_only
  def testRunMetadata_wholeRunMetadata(self):
    # [FUNCTION DOC]
    # Asserts the logged tensor deserializes to the full run metadata (step stats and function graphs).
    expected_run_metadata = """
      step_stats {
        dev_stats {
          device: "cpu:0"
          node_stats {
            node_name: "hello"
          }
        }
      }
      function_graphs {
        pre_optimization_graph {
          node {
            name: "foo"
          }
        }
      }
    """
    meta = self.create_run_metadata()
    event = self.run_metadata(name='my_name', data=meta, step=1)
    first_val = event.summary.value[0]

    actual_run_metadata = config_pb2.RunMetadata.FromString(
        first_val.tensor.string_val[0])
    self.assertProtoEquals(expected_run_metadata, actual_run_metadata)

  @test_util.run_v2_only
  def testRunMetadata_usesDefaultStep(self):
    # [FUNCTION DOC]
    # With the global default step 42, logs run metadata without a step and asserts the event step is
    # 42. Resets the global step.
    meta = config_pb2.RunMetadata()
    try:
      summary_ops.set_step(42)
      event = self.run_metadata(name='my_name', data=meta)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  @test_util.run_v2_only
  def testRunMetadataGraph_usesNameAsTag(self):
    # [FUNCTION DOC]
    # Same as the name-as-tag test for the graphs-only function.
    meta = config_pb2.RunMetadata()

    with ops.name_scope('foo', skip_on_eager=False):
      event = self.run_metadata_graphs(name='my_name', data=meta, step=1)
      first_val = event.summary.value[0]

    self.assertEqual('foo/my_name', first_val.tag)

  @test_util.run_v2_only
  def testRunMetadataGraph_summaryMetadata(self):
    # [FUNCTION DOC]
    # Asserts the metadata plugin name is the run-metadata graph plugin with content 1.
    expected_summary_metadata = """
      plugin_data {
        plugin_name: "graph_run_metadata_graph"
        content: "1"
      }
    """
    meta = config_pb2.RunMetadata()
    event = self.run_metadata_graphs(name='my_name', data=meta, step=1)
    actual_summary_metadata = event.summary.value[0].metadata
    self.assertProtoEquals(expected_summary_metadata, actual_summary_metadata)

  @test_util.run_v2_only
  def testRunMetadataGraph_runMetadataFragment(self):
    # [FUNCTION DOC]
    # Asserts the logged run metadata contains only the function graphs, not the step stats.
    expected_run_metadata = """
      function_graphs {
        pre_optimization_graph {
          node {
            name: "foo"
          }
        }
      }
    """
    meta = self.create_run_metadata()

    event = self.run_metadata_graphs(name='my_name', data=meta, step=1)
    first_val = event.summary.value[0]

    actual_run_metadata = config_pb2.RunMetadata.FromString(
        first_val.tensor.string_val[0])
    self.assertProtoEquals(expected_run_metadata, actual_run_metadata)

  @test_util.run_v2_only
  def testRunMetadataGraph_usesDefaultStep(self):
    # [FUNCTION DOC]
    # Uses the global default step 42 for graphs-only run metadata; asserts the event step is 42 and
    # resets the global step.
    meta = config_pb2.RunMetadata()
    try:
      summary_ops.set_step(42)
      event = self.run_metadata_graphs(name='my_name', data=meta)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  @test_util.run_v2_only
  def testTrace(self):
    # [FUNCTION DOC]
    # Traces a small compiled function, exports it and asserts the logged run metadata has a function
    # graphs field.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Computes two constants raised to a power and returns the result.
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    event = self.run_trace(f)

    first_val = event.summary.value[0]
    actual_run_metadata = config_pb2.RunMetadata.FromString(
        first_val.tensor.string_val[0])

    # Content of function_graphs is large and, for instance, device can change.
    self.assertTrue(hasattr(actual_run_metadata, 'function_graphs'))

  @test_util.run_v2_only
  def testTrace_cannotEnableTraceInFunction(self):
    # [FUNCTION DOC]
    # Enabling trace inside a compiled function must log a warning mentioning that it cannot be done
    # inside a function. Asserts on a mocked logger.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Tries to turn tracing on, then returns a computed value.
      summary_ops.trace_on(graph=True, profiler=False)
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    with test.mock.patch.object(logging, 'warn') as mock_log:
      f()
      self.assertRegex(
          str(mock_log.call_args), 'Cannot enable trace inside a tf.function.')

  @test_util.run_v2_only
  def testTrace_cannotEnableTraceInGraphMode(self):
    # [FUNCTION DOC]
    # Enabling trace in graph mode must warn that it needs eager mode.
    with test.mock.patch.object(logging, 'warn') as mock_log:
      with context.graph_mode():
        summary_ops.trace_on(graph=True, profiler=False)
      self.assertRegex(
          str(mock_log.call_args), 'Must enable trace in eager mode.')

  @test_util.run_v2_only
  def testTrace_cannotExportTraceWithoutTrace(self):
    # [FUNCTION DOC]
    # Exporting without having enabled trace must raise a value error.
    with self.assertRaisesRegex(ValueError, 'Must enable trace before export.'):
      summary_ops.trace_export(name='foo', step=1)

  @test_util.run_v2_only
  def testTrace_cannotExportTraceInFunction(self):
    # [FUNCTION DOC]
    # After enabling trace, exporting inside a compiled function must log a warning that it cannot be
    # done inside a function.
    summary_ops.trace_on(graph=True, profiler=False)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Tries to export the trace and returns a computed value.
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      summary_ops.trace_export(name='foo', step=1)
      return x**y

    with test.mock.patch.object(logging, 'warn') as mock_log:
      f()
      self.assertRegex(
          str(mock_log.call_args), 'Cannot export trace inside a tf.function.')

  @test_util.run_v2_only
  def testTrace_cannotExportTraceInGraphMode(self):
    # [FUNCTION DOC]
    # Exporting in graph mode must log a warning about needing eager execution.
    with test.mock.patch.object(logging, 'warn') as mock_log:
      with context.graph_mode():
        summary_ops.trace_export(name='foo', step=1)
      self.assertRegex(
          str(mock_log.call_args),
          'Can only export trace while executing eagerly.')

  @test_util.run_v2_only
  def testTrace_usesDefaultStep(self):
    # [FUNCTION DOC]
    # With default step 42 and a null step argument, the exported trace event has step 42. Resets the
    # global step.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Computes two constants raised to a power and returns the result.
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    try:
      summary_ops.set_step(42)
      event = self.run_trace(f, step=None)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  @test_util.run_v2_only
  def testTrace_withProfiler(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test has no assertions; it only checks that tracing with the profiler enabled
    # and exporting does not raise.
    # Enables tracing with graph and profiler output in a temporary directory, runs a function under a
    # writer and exports the trace.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Computes two constants raised to a power and returns the result.
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer_v2(logdir)
    profiler_outdir = self.get_temp_dir()
    summary_ops.trace_on(
        graph=True, profiler=True, profiler_outdir=profiler_outdir
    )
    with writer.as_default():
      f()
      summary_ops.trace_export(name='foo', step=1)
    writer.close()

  @test_util.run_v2_only
  def testGraph_graph(self):
    # [FUNCTION DOC]
    # Exports a function's graph through the graph summary function and asserts the event holds a
    # graph definition.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Computes two constants raised to a power and returns the result.
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    def summary_op_fn():
      # [FUNCTION DOC]
      # HELPER: callback for the enclosing test.
      # Writes the graph summary for the concrete function's graph.
      summary_ops.graph(f.get_concrete_function().graph)

    event = self.exec_summary_op(summary_op_fn)
    self.assertIsNotNone(event.graph_def)

  @test_util.run_v2_only
  def testGraph_graphDef(self):
    # [FUNCTION DOC]
    # Same as the graph test but passing a graph definition message.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Computes two constants raised to a power and returns the result.
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    def summary_op_fn():
      # [FUNCTION DOC]
      # HELPER: callback for the enclosing test.
      # Writes the graph summary using the graph definition of the concrete function.
      summary_ops.graph(f.get_concrete_function().graph.as_graph_def())

    event = self.exec_summary_op(summary_op_fn)
    self.assertIsNotNone(event.graph_def)

  @test_util.run_v2_only
  def testGraph_invalidData(self):
    # [FUNCTION DOC]
    # Passing a string to the graph summary function must raise a value error saying it is neither a
    # graph nor a graph definition.
    def summary_op_fn():
      # [FUNCTION DOC]
      # HELPER: callback for the enclosing test.
      # Calls the graph summary function with an invalid string argument.
      summary_ops.graph('hello')

    with self.assertRaisesRegex(
        ValueError,
        r'\'graph_data\' is not tf.Graph or tf.compat.v1.GraphDef',
    ):
      self.exec_summary_op(summary_op_fn)

  @test_util.run_v2_only
  def testGraph_fromGraphMode(self):
    # [FUNCTION DOC]
    # Calling the graph summary function from a compiled function must raise a value error saying it
    # cannot be invoked inside a graph context.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function body for the enclosing test.
      # Computes two constants raised to a power and returns the result.
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    @def_function.function
    def g(graph):
      # [FUNCTION DOC]
      # HELPER: compiled function used by the enclosing test.
      # Takes a graph argument and calls the graph summary function with it.
      summary_ops.graph(graph)

    def summary_op_fn():
      # [FUNCTION DOC]
      # HELPER: callback for the enclosing test.
      # Serializes the graph definition of f (with shapes) into a string tensor and passes it to g.
      graph_def = f.get_concrete_function().graph.as_graph_def(add_shapes=True)
      func_graph = constant_op.constant(graph_def.SerializeToString())
      g(func_graph)

    with self.assertRaisesRegex(
        ValueError,
        r'graph\(\) cannot be invoked inside a graph context.',
    ):
      self.exec_summary_op(summary_op_fn)


def events_from_file(filepath):
  # [FUNCTION DOC]
  # HELPER: shared test utility.
  # Reads every record from the event file at the given path, parses each as an event message, and
  # returns the list of events.
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.Event protos in the event file.
  """
  records = list(tf_record.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def events_from_logdir(logdir):
  # [FUNCTION DOC]
  # HELPER: shared test utility.
  # Asserts the directory exists and contains exactly one file, then returns all events in it.
  # Raises an assertion error otherwise.
  """Returns all events in the single eventfile in logdir.

  Args:
    logdir: The directory in which the single event file is sought.

  Returns:
    A list of all tf.Event protos from the single event file.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert gfile.Exists(logdir)
  files = gfile.ListDirectory(logdir)
  assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
  return events_from_file(os.path.join(logdir, files[0]))


def events_from_multifile_logdir(logdir):
  # [FUNCTION DOC]
  # HELPER: shared test utility.
  # KNOWN ISSUE: the docstring's Raises section claims an assertion error when the directory does not
  # hold exactly one file, but the code only asserts that the directory exists and accepts any number.
  # Returns a mapping from file name to event list for all files whose names contain the event-file
  # marker in the directory.
  """Returns map of filename to events for all `tfevents` files in the logdir.

  Args:
    logdir: The directory from which to load events.

  Returns:
    A dict mapping from relative filenames to lists of tf.Event protos.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert gfile.Exists(logdir)
  files = [file for file in gfile.ListDirectory(logdir) if 'tfevents' in file]
  return {file: events_from_file(os.path.join(logdir, file)) for file in files}


def to_numpy(summary_value):
  # [FUNCTION DOC]
  # HELPER: shared test utility.
  # Converts the tensor of a summary value message into a numpy array and returns it.
  return tensor_util.MakeNdarray(summary_value.tensor)


if __name__ == '__main__':
  test.main()
