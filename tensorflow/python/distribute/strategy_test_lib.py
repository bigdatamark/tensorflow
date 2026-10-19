# [MODULE DOC]
# HELPER: helper module providing a library for testing DistributionStrategy implementations.
# Contains private exception-raising callbacks (including nested merge-call variants), small
# utilities (event reading, Keras-like variable creation, strategy type predicates, optional
# traced-function wrapping, sum/mean all-reduce callbacks) and base test-case classes with
# reusable test bodies: a generic base, one-device and two-device bases, and a remote
# single-worker base. Concrete strategy tests inherit these and call the underscore-prefixed
# methods. No import-time side effects.
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
"""Library for testing DistributionStrategy descendants."""

import functools
import os
import tempfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as mwms_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


class _TestException(Exception):
  # [CLASS DOC]
  # HELPER: private exception type raised by the helper functions in this module so tests can
  # check that an error raised inside replica or merge calls propagates unchanged to the caller.
  pass


# Conditionally wrap the fn in a def_function.function (so it runs in graph
# mode).
def _maybe_run_in_function(fn, run_in_function=False):
  # [FUNCTION DOC]
  # HELPER: lets a test optionally run a callback as a traced graph function.
  # Inputs: fn (callback) and run_in_function (boolean, default false). If the flag is off, or
  # execution is not eager, returns fn unchanged; otherwise returns fn wrapped as a traced
  # function. No side effects.
  if not run_in_function or not context.executing_eagerly():
    return fn
  else:
    return def_function.function()(fn)


# May be the argument to either distribution.extended.call_for_each_replica() or
# get_replica_context().merge_call()
def _raise_exception_fn(_=None):
  # [FUNCTION DOC]
  # HELPER: test callback that always raises the private test exception. Accepts and ignores an
  # optional single argument so it can be used both as a per-replica function and as a merge
  # function.
  raise _TestException()


# Must be the argument to a distribution.extended.call_for_each_replica() call,
# calls a get_replica_context().merge_call() that raises an exception.
def _merge_raises_fn():
  # [FUNCTION DOC]
  # HELPER: per-replica test callback. Gets the current replica context and performs a merge
  # call whose merge function raises the private test exception. Meant to be passed to the
  # strategy's call-for-each-replica. Takes nothing; always raises.
  distribute_lib.get_replica_context().merge_call(_raise_exception_fn)


# Must be the argument to a get_replica_context().merge_call() call, calls
# dist.extended.call_for_each_replica() with a function that raises an
# exception.
def _call_raises_fn(dist):
  # [FUNCTION DOC]
  # HELPER: merge-function test callback. Input: dist, the strategy. Calls a function that
  # raises the private test exception on each replica. Always raises. Meant to be passed to a
  # replica context merge call.
  dist.extended.call_for_each_replica(_raise_exception_fn)


# Must be the argument to a distribution.extended.call_for_each_replica() call,
# calls a get_replica_context().merge_call() that calls a
# call_for_each_replica() that raises an exception.
def _merge_call_raises_fn():
  # [FUNCTION DOC]
  # HELPER: per-replica test callback. Performs a merge call whose merge function in turn calls
  # a per-replica function that raises the private test exception. Takes nothing; always raises.
  distribute_lib.get_replica_context().merge_call(_call_raises_fn)


# Must be the argument to a get_replica_context().merge_call() call, calls
# dist.extended.call_for_each_replica() with a function that calls a
# get_replica_context().merge_call() that raises an exception.
def _call_merge_raises_fn(dist):
  # [FUNCTION DOC]
  # HELPER: merge-function test callback. Input: dist, the strategy. Calls, on each replica, a
  # function that does a merge call raising the private test exception. Always raises.
  dist.extended.call_for_each_replica(_merge_raises_fn)


# Must be the argument to a distribution.extended.call_for_each_replica() call,
# calls a get_replica_context().merge_call() that calls a
# call_for_each_replica() that calls a get_replica_context().merge_call() that
# raises an exception.
def _merge_call_merge_raises_fn():
  # [FUNCTION DOC]
  # HELPER: per-replica test callback. Performs a merge call whose merge function calls
  # per-replica code that itself does a merge call raising the private test exception (the
  # deepest nesting of the exception helpers). Takes nothing; always raises.
  distribute_lib.get_replica_context().merge_call(_call_merge_raises_fn)


def _events_from_logdir(test_case, logdir):
  # [FUNCTION DOC]
  # HELPER: reads summary events written by a summary file writer.
  # Inputs: test_case (the running test case, used for assertions) and logdir (directory path).
  # Asserts the directory exists and contains exactly one file, reads all records from that
  # file and parses each as an event message. Returns the list of events in file order.
  # Fails the test through assertions if the directory is missing or holds not exactly one file.
  """Reads summary events from log directory."""
  test_case.assertTrue(gfile.Exists(logdir))
  files = gfile.ListDirectory(logdir)
  test_case.assertLen(files, 1)
  records = list(tf_record.tf_record_iterator(os.path.join(logdir, files[0])))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def create_variable_like_keras_layer(name, shape, dtype):
  # [FUNCTION DOC]
  # HELPER: shared test utility that creates a trainable variable the way a Keras layer would,
  # i.e. with a deferred initial value from a Glorot-uniform initializer bound to the given
  # shape and dtype. Inputs: name (string), shape (tuple of ints), dtype (data type). Returns
  # the new variable (created under whatever strategy scope is active).
  """Utility for create variables that works like variable in keras layer."""
  initializer = functools.partial(
      init_ops_v2.GlorotUniform(), shape, dtype=dtype)
  return variables.Variable(
      initial_value=initializer, name=name, trainable=True)


def is_optimizer_v2_instance(optimizer_obj):
  # [FUNCTION DOC]
  # HELPER: test utility telling the optimizer implementation generation apart.
  # KNOWN ISSUE: no docstring; relies on the minimize method having at least one default
  # argument. If it has none, the slice of required arguments becomes empty (a negative zero
  # slice) and the result is always false; if the defaults are nothing/null the length call
  # raises a type error.
  # Input: an optimizer object. Inspects the signature of its minimize method and returns true
  # if var_list is among the required (non-default) arguments, which marks the v2 optimizer API.
  # For a optimizer instance, the v2 implementation has var_list as a required
  # argument.
  arg_spec = tf_inspect.getfullargspec(optimizer_obj.minimize)
  return "var_list" in arg_spec.args[:-len(arg_spec.defaults)]


def is_mirrored_strategy(strategy: distribute_lib.Strategy) -> bool:
  # [FUNCTION DOC]
  # HELPER: test utility. Returns true if the strategy is an instance of the mirrored strategy
  # class (V2 or V1 flavour), else false.
  return isinstance(
      strategy,
      (mirrored_lib.MirroredStrategy, mirrored_lib.MirroredStrategyV1))


def is_multi_worker_mirrored_strategy(
    strategy: distribute_lib.Strategy) -> bool:
  # [FUNCTION DOC]
  # HELPER: test utility. Returns true if the strategy is an instance of the multi-worker
  # (collective all-reduce) strategy class, V2 or V1 flavour, else false.
  return isinstance(strategy, (mwms_lib.CollectiveAllReduceStrategy,
                               mwms_lib.CollectiveAllReduceStrategyV1))


def is_tpu_strategy(strategy: distribute_lib.Strategy) -> bool:
  # [FUNCTION DOC]
  # HELPER: test utility. Returns true if the strategy is an instance of one of the TPU strategy
  # classes (original, V1 or V2), else false. Duplicates the same helper in the sibling test
  # utility module.
  return isinstance(strategy,
                    (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1,
                     tpu_strategy.TPUStrategyV2))


class DistributionTestBase(test.TestCase):
  # [CLASS DOC]
  # Base test case offering reusable test bodies that should pass for any distribution strategy
  # (the underscore-prefixed test methods are called from concrete test classes with a strategy
  # instance): loss minimization in eager and graph mode, summary writing from only replica zero,
  # replica ids, exception propagation, input-function and dataset iteration checks, global step
  # updates, numpy-dataset input, and trainable-flag semantics. Holds no state of its own.
  """Some tests that should work with any DistributionStrategy."""

  def _test_minimize_loss_eager(self, d):
    # [FUNCTION DOC]
    # HELPER: shared test body (eager mode). Input: d, the strategy under test. In its scope,
    # creates a one-by-one kernel variable and defines a squared-error loss against the constant
    # one. Builds an implicit-gradient function and runs ten steps; each step computes per-replica
    # gradients, reads each variable, sums the gradient across replicas to the variable's
    # destinations and updates the variable by subtracting a fifth of the gradient. Asserts that
    # the absolute error after the last step is smaller than the error before the first step.
    with d.scope():
      kernel = create_variable_like_keras_layer(
          name="kernel", shape=(1, 1), dtype=dtypes.float32)
      def loss(x):
        # [FUNCTION DOC]
        # Squared error of the product of input and kernel against the constant one.
        y = array_ops.reshape(
            math_ops.mat_mul(x, kernel), []) - array_ops.identity(1.)
        return y * y
      # TODO(isaprykin): Extract implicit_grad+get_filtered_grad_fn into a
      # common `implicit_grad` function and put it in DistributionStrategy.
      grad_fn = backprop.implicit_grad(loss)
      grad_fn = optimizer.get_filtered_grad_fn(grad_fn)

      def update(v, g):
        # [FUNCTION DOC]
        # Update callback: subtracts one fifth of the gradient g from variable v and returns the
        # update.
        return v.assign_sub(0.2 * g)

      one = array_ops.identity([[1.]])

      def step():
        # [FUNCTION DOC]
        # Performs one optimization step eagerly: per-replica gradients, then for each
        # gradient/variable pair reads the variable, reduces the gradient by sum to the variable,
        # applies the update and reads the variable again. Returns two lists: values before and
        # after the update.
        """Perform one optimization step."""
        # Run forward & backward to get gradients, variables list.
        g_v = d.extended.call_for_each_replica(grad_fn, args=(one,))

        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = d.extended.read_var(v)
          before_list.append(fetched)
          # control_dependencies irrelevant but harmless in eager execution
          with ops.control_dependencies([fetched]):
            g = d.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(
                d.extended.update(v, update, args=(g,), group=False)):
              after_list.append(d.extended.read_var(v))
        return before_list, after_list

      for i in range(10):
        b, a = step()
        if i == 0:
          before, = b  # pylint: disable=unbalanced-tuple-unpacking
        after, = a  # pylint: disable=unbalanced-tuple-unpacking

      error_before = abs(before.numpy() - 1)
      error_after = abs(after.numpy() - 1)
      # Error should go down
      self.assertLess(error_after, error_before)

  def _test_minimize_loss_graph(self,
                                d,
                                soft_placement=False,
                                learning_rate=0.2):
    # [FUNCTION DOC]
    # HELPER: shared test body (graph mode). Inputs: d (strategy), soft_placement (boolean,
    # default false; allows soft device placement in the session config) and learning_rate (float,
    # default 0.2). Limits per-process GPU memory to thirty percent, enters graph mode with a new
    # graph, a cached session and the strategy scope, then builds the same one-by-one kernel
    # squared-error problem as the eager variant and one update step. Initializes variables and
    # runs the step ten times in the session, recording the first before-value and last
    # after-value. Asserts the error decreased.
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = soft_placement
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with context.graph_mode(), \
         ops.Graph().as_default(), \
         self.cached_session(config=config) as sess, \
         d.scope():
      kernel = create_variable_like_keras_layer(
          name="kernel", shape=(1, 1), dtype=dtypes.float32)

      def loss(x):
        # [FUNCTION DOC]
        # Squared error of the product of input and kernel against the constant one.
        y = array_ops.reshape(
            math_ops.mat_mul(x, kernel), []) - array_ops.identity(1.)
        return y * y

      grad_fn = backprop.implicit_grad(loss)

      def update(v, g):
        # [FUNCTION DOC]
        # Update callback: subtracts the learning rate times the gradient g from variable v and
        # returns the update.
        return v.assign_sub(learning_rate * g)

      one = array_ops.identity([[1.]])

      def step():
        # [FUNCTION DOC]
        # Builds one optimization step (graph form): per-replica gradients, then for each
        # gradient/variable pair reads the variable, reduces the gradient by sum to the variable,
        # applies the update ordered after the read and reads the variable again. Returns two
        # lists: values before and after the update.
        """Perform one optimization step."""
        # Run forward & backward to get gradients, variables list.
        g_v = d.extended.call_for_each_replica(grad_fn, args=(one,))

        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = d.extended.read_var(v)
          before_list.append(fetched)
          with ops.control_dependencies([fetched]):
            g = d.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(
                d.extended.update(v, update, args=(g,), group=False)):
              after_list.append(d.extended.read_var(v))
        return before_list, after_list

      before_out, after_out = step()
      variables.global_variables_initializer().run()
      for i in range(10):
        b, a = sess.run((before_out, after_out))
        if i == 0:
          before, = b
        after, = a

      error_before = abs(before - 1)
      error_after = abs(after - 1)
      # Error should go down
      self.assertLess(error_after, error_before)

  def _test_summary_for_replica_zero_only(self, d):
    # [FUNCTION DOC]
    # HELPER: shared test body. Input: d, the strategy. Creates a temporary log directory and, in
    # the strategy scope with always-record summaries, a global step and a file summary writer.
    # Runs a per-replica function that writes a scalar summary named a with the replica id as the
    # value. In graph mode also initializes the writer, runs the outputs and closes the writer.
    # Reads events back: expects exactly two (file header plus one record) and that the second
    # has tag a and value zero, proving only replica zero wrote. Side effect: creates a temporary
    # directory that is never removed.
    logdir = tempfile.mkdtemp()

    def run_fn():
      # [FUNCTION DOC]
      # Per-replica function: inside the writer's default context, writes a scalar summary named
      # a whose value is the replica id in the sync group, and returns the write result.
      """Function executed for each replica."""
      with summary_writer.as_default():
        replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
        return summary_ops.write("a", replica_id)

    with self.cached_session() as sess, d.scope(), \
        summary_ops.always_record_summaries():
      # We need global_step because summary writing op *always* has global_step
      # as input, even when we always record summary or never record summary.
      global_step = training_util.get_or_create_global_step()
      if not context.executing_eagerly():
        # When executing eagerly, variables are initialized immediately after
        # creation, and its initializer will be None.
        global_step.initializer.run()
      summary_ops.set_step(0)
      summary_writer = summary_ops.create_file_writer(logdir)
      output = d.extended.call_for_each_replica(run_fn)
      unwrapped = d.unwrap(output)
      if not context.executing_eagerly():
        sess.run(summary_writer.init())
        sess.run(unwrapped)
        sess.run(summary_writer.close())

      events = _events_from_logdir(self, logdir)
      # There will be 2 entries: 1 summary file header entry, and 1 entry
      # written by replica 0.
      self.assertLen(events, 2)
      self.assertEqual(events[1].summary.value[0].tag, "a")
      self.assertEqual(events[1].summary.value[0].simple_value, 0.0)

  def _test_replica_id(self, d):
    # [FUNCTION DOC]
    # HELPER: shared test body. Input: d, the strategy. Calls a function on every replica that
    # evaluates the replica id, asserts it is below the number of worker devices and not seen
    # before, and marks it. Finally asserts every worker device index was marked exactly once.
    with d.scope():
      expected_devices = [False] * len(d.extended.worker_devices)

      def mark_devices_fn():
        # [FUNCTION DOC]
        # Per-replica callback: evaluates the replica id, asserts it is a valid, so far unused
        # index, and marks that index as seen in the enclosing list.
        replica_id = self.evaluate(
            distribute_lib.get_replica_context().replica_id_in_sync_group)
        self.assertLess(replica_id, len(d.extended.worker_devices))
        self.assertFalse(expected_devices[replica_id])
        expected_devices[replica_id] = True

      d.extended.call_for_each_replica(mark_devices_fn)
      self.assertAllEqual(expected_devices,
                          [True] * len(d.extended.worker_devices))

  def _test_call_and_merge_exceptions(self, dist):
    # [FUNCTION DOC]
    # HELPER: shared test body. Input: dist, the strategy. In its scope, calls per-replica
    # callbacks that raise the private test exception directly, via a merge call, via a merge
    # call that calls per-replica code, and via nested merge/replica/merge calls. Asserts the test
    # exception propagates in all four cases.
    with dist.scope():
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_raise_exception_fn)
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_merge_raises_fn)
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_merge_call_raises_fn)
      with self.assertRaises(_TestException):
        dist.extended.call_for_each_replica(_merge_call_merge_raises_fn)

  def _input_fn_to_test_input_context(self, dataset_or_callable_fn,
                                      expected_num_replicas_in_sync,
                                      expected_num_input_pipelines,
                                      expected_input_pipeline_id):
    # [FUNCTION DOC]
    # HELPER: builds an input function that verifies the input context it receives. Inputs:
    # dataset_or_callable_fn (callback producing a dataset or an element callable),
    # expected_num_replicas_in_sync and expected_num_input_pipelines (integers), and
    # expected_input_pipeline_id (integer, or nothing/null to mean it should equal the running
    # count of invocations, as in in-graph replication). Returns the checking input function.
    # Use a list of one element as counter so that it can be captured by the
    # `_input_fn`. This counter is incremented by 1 each time an input_fn is
    # called. We use this counter to check whether the `input_pipeline_id`
    # matches the counter in the in-graph replication.
    worker_id_counter = [0]

    def _input_fn(input_context):
      # [FUNCTION DOC]
      # Input function: asserts the given input context is present, that its replica and
      # input-pipeline counts match expectations, and that the pipeline id matches the expected
      # id or the call counter (which is then incremented). Returns what the wrapped dataset or
      # callable function produces.
      """Input fn for testing."""
      self.assertIsNotNone(input_context)
      self.assertEqual(expected_num_replicas_in_sync,
                       input_context.num_replicas_in_sync)
      self.assertEqual(expected_num_input_pipelines,
                       input_context.num_input_pipelines)
      if expected_input_pipeline_id is not None:
        self.assertEqual(expected_input_pipeline_id,
                         input_context.input_pipeline_id)
      else:
        self.assertEqual(worker_id_counter[0], input_context.input_pipeline_id)
        worker_id_counter[0] += 1

      return dataset_or_callable_fn()

    return _input_fn

  def _test_input_fn_iterable(
      self, strategy, input_fn, expected_values, ignore_order=False):
    # [FUNCTION DOC]
    # HELPER: shared test body for datasets distributed from an input function. Inputs: strategy,
    # input_fn (input function), expected_values (list of per-step expected lists of per-replica
    # values) and ignore_order (boolean, default false: compare order-insensitively). Distributes
    # the dataset. In eager mode iterates it, comparing each step's local results with the
    # expected values, checks that a stop-iteration error follows, then creates a new iterator and
    # repeats the comparison. In graph mode creates an initializable iterator and delegates to
    # the iterator check helper with reinitialization enabled.
    assert_same = self.assertCountEqual if ignore_order else self.assertEqual

    iterable = strategy.distribute_datasets_from_function(input_fn)
    if context.executing_eagerly():
      iterator = iter(iterable)

      for expected_value in expected_values:
        computed_value = self.evaluate(
            list(strategy.experimental_local_results(next(iterator))))
        assert_same(expected_value, computed_value)

      with self.assertRaises(StopIteration):
        self.evaluate(strategy.experimental_local_results(next(iterator)))

      # After re-initializing the iterator, should be able to iterate again.
      iterator = iter(iterable)

      for expected_value in expected_values:
        computed_value = self.evaluate(
            list(strategy.experimental_local_results(next(iterator))))
        assert_same(expected_value, computed_value)
    else:
      iterator = dataset_ops.make_initializable_iterator(iterable)
      self._test_input_fn_iterator(iterator, strategy.extended.worker_devices,
                                   expected_values, test_reinitialize=True,
                                   ignore_order=ignore_order)

  def _test_input_fn_iterator(self,
                              iterator,
                              devices,
                              expected_values,
                              sess=None,
                              test_reinitialize=True,
                              ignore_order=False):
    # [FUNCTION DOC]
    # HELPER: shared test body for iterators. Inputs: iterator, devices (list of device names),
    # expected_values (list of per-step lists of per-replica values), sess (optional session; if
    # absent uses the test case's evaluate), test_reinitialize (boolean, default true) and
    # ignore_order (boolean, default false). Initializes the iterator, and for each expected step
    # fetches the next element, selects each replica's component and compares with the expected
    # values (order-insensitive if requested). Asserts that fetching one more element raises an
    # out-of-range error. If reinitialization is on, reinitializes and repeats the full check.
    evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
    evaluate(iterator.initializer)

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [distribute_utils.select_replica(r, next_element) for r in
           range(len(devices))])
      if ignore_order:
        self.assertCountEqual(expected_value, computed_value)
      else:
        self.assertEqual(expected_value, computed_value)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next()
      evaluate(
          [distribute_utils.select_replica(r, next_element) for r in
           range(len(devices))])

    # After re-initializing the iterator, should be able to iterate again.
    if test_reinitialize:
      evaluate(iterator.initializer)

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = evaluate([
            distribute_utils.select_replica(r, next_element) for r in
            range(len(devices))
        ])
        if ignore_order:
          self.assertCountEqual(expected_value, computed_value)
        else:
          self.assertEqual(expected_value, computed_value)

  def _test_global_step_update(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body. Input: strategy. In its scope creates a non-trainable integer
    # scalar global step variable aggregated by taking the first replica, initializes variables,
    # and runs a per-replica function that increments it and reads it. Groups and evaluates the
    # update operations, evaluates the local results of the read value, and asserts every
    # replica saw the value one.
    with strategy.scope():
      global_step = variable_scope.get_variable(
          "global_step",
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        # [FUNCTION DOC]
        # Per-replica function: increments the global step by one and reads its value. Returns
        # the pair of the increment operation and the read value.
        train_op = global_step.assign_add(1)
        value = global_step.read_value()
        return train_op, value

      train_ops, value = strategy.extended.call_for_each_replica(model_fn)
      self.evaluate(strategy.group(train_ops))
      global_step_tensors = strategy.experimental_local_results(value)
      global_step_values = self.evaluate(global_step_tensors)
      self.assertEqual((1,) * len(global_step_tensors), global_step_values)

  def _test_numpy_dataset(self, strategy, session=None, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body for numpy-array datasets (V1 strategies only; skipped otherwise).
    # Inputs: strategy, session (optional session to use instead of a cached one) and
    # run_in_function (boolean, default false; wraps the per-replica step in a traced function).
    # Builds a six-example dataset from two arrays, repeats it twice, batches it (global or
    # per-replica batch size depending on the strategy; dropping the remainder if static shapes
    # are required) and makes a dataset iterator. Runs and concatenates local results across
    # replicas: the first two passes must equal the original arrays and a third must raise an
    # out-of-range error.
    if not isinstance(strategy, distribute_lib.StrategyV1):
      self.skipTest("n/a: V1 only")
    cached_session = session or self.cached_session()
    with strategy.scope(), cached_session as sess:
      x = np.asarray([[1, 2], [6, 12], [2, 4], [5, 10], [3, 6], [4, 8]])
      y = np.asarray([5, 4, 3, 2, 1, 0])
      batch_size = 6
      if not strategy.extended._global_batch_size:  # pylint: disable=protected-access
        batch_size = batch_size // strategy.num_replicas_in_sync

      ds = strategy.extended.experimental_make_numpy_dataset(
          (x, y), session=sess or self.cached_session())
      ds = ds.repeat(2)  # 2 epochs
      # We need to use the drop_remainder argument to get a known static
      # input shape which is required for TPUs.
      drop_remainder = strategy.extended.experimental_require_static_shapes
      ds = ds.batch(batch_size, drop_remainder=drop_remainder)
      i = strategy.make_dataset_iterator(ds)

      self.evaluate(i.initializer)

      def run_and_concatenate(strategy, i):
        # [FUNCTION DOC]
        # Runs an identity step on the iterator's next batch across replicas, evaluates the local
        # results of both elements and returns them concatenated across replicas as two arrays.
        x, y = strategy.experimental_run(
            _maybe_run_in_function(lambda z: z, run_in_function), i)
        x, y = self.evaluate((strategy.experimental_local_results(x),
                              strategy.experimental_local_results(y)))
        return np.concatenate(x), np.concatenate(y)

      x_1, y_1 = run_and_concatenate(strategy, i)
      self.assertAllEqual(x, x_1)
      self.assertAllEqual(y, y_1)
      x_2, y_2 = run_and_concatenate(strategy, i)
      self.assertAllEqual(x, x_2)
      self.assertAllEqual(y, y_2)
      with self.assertRaises(errors.OutOfRangeError):
        run_and_concatenate(strategy, i)

  def _test_trainable_variable(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body. Input: strategy. For both the V1 and the standard variable
    # classes, within the strategy scope, asserts: a default variable is trainable; a variable
    # synchronized on read defaults to not trainable; and explicit trainable flags are honored
    # even with on-read synchronization.
    for cls in [variable_v1.VariableV1, variables.Variable]:
      with strategy.scope():
        v1 = cls(1.0)
        self.assertEqual(True, v1.trainable)

        v2 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ)
        self.assertEqual(False, v2.trainable)

        v3 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ,
                 trainable=True)
        self.assertEqual(True, v3.trainable)

        v4 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ,
                 trainable=False)
        self.assertEqual(False, v4.trainable)


class OneDeviceDistributionTestBase(test.TestCase):
  # [CLASS DOC]
  # Base test case with reusable test bodies for strategies that have exactly one device: run
  # semantics, all-reduce sum and mean (values, graph gradients, tape gradients) and colocation
  # of the device and its input device in a local cluster. Concrete tests call the underscore
  # methods with a strategy.
  """Some tests that should work with any one-device DistributionStrategy."""

  def _test_run(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body. Runs a function returning four on the strategy and checks the
    # unwrapped result; runs a function producing a mapping with double and square of the
    # previous output (expects eight and sixteen); then runs a function taking keyword arguments
    # (a plus two times b plus two) fed from that mapping (expects forty-two).
    out1 = strategy.run(lambda: array_ops.identity(4.))
    self.assertAllEqual([4.], self.evaluate(strategy.unwrap(out1)))

    out2 = strategy.run(lambda x: {"a": x * 2, "b": x * x}, args=(out1,))
    out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
    self.assertAllEqual([8.], out2_vals["a"])
    self.assertAllEqual([16.], out2_vals["b"])

    out3 = strategy.run(lambda b, a: a + 2 * b + 2, kwargs=out2)
    self.assertAllEqual([42.], self.evaluate(strategy.unwrap(out3)))

  def _test_all_reduce_sum(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body: all-reduce sum with a single replica must return its inputs
    # unchanged (scalar four and a two-element vector).
    self._test_collective_comms(
        strategy, _all_sum, inputs=(4., [42., 43.]), expected=(4., [42., 43.]))

  def _test_all_reduce_sum_gradients(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body: graph-mode gradient of an all-reduce sum with one replica and
    # input multiplier four must be four.
    self._test_collective_comms_gradients(
        strategy, _all_sum, inputs=[4.], expected_grads=[4.])

  def _test_all_reduce_sum_gradient_tape(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body: tape-based gradient of an all-reduce sum with one replica and
    # input multiplier four must be four.
    self._test_collective_comms_gradient_tape(
        strategy, _all_sum, inputs=[4.], expected_grads=[4.])

  def _test_all_reduce_mean(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body: all-reduce mean with a single replica must return its inputs
    # unchanged (scalar two and a two-element vector).
    self._test_collective_comms(
        strategy, _all_mean, inputs=(2., [21., 22.]), expected=(2., [21., 22.]))

  def _test_all_reduce_mean_gradients(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body: graph-mode gradient of an all-reduce mean with one replica and
    # input multiplier five must be five.
    self._test_collective_comms_gradients(
        strategy, _all_mean, inputs=[5.], expected_grads=[5.])

  def _test_all_reduce_mean_gradient_tape(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body: tape-based gradient of an all-reduce mean with one replica and
    # input multiplier five must be five.
    self._test_collective_comms_gradient_tape(
        strategy, _all_mean, inputs=[5.], expected_grads=[5.])

  def _test_collective_comms(self, strategy, comm_fn, inputs, expected):
    # [FUNCTION DOC]
    # HELPER: shared test body. Inputs: strategy, comm_fn (collective callback), inputs (tuple of
    # a scalar and a vector) and expected (tuple of expected scalar and vector).
    # KNOWN ISSUE: the name inputs is rebound to the iterator while the dataset-producing lambda
    # still refers to it by name, so the lambda only works because it is called before rebinding.
    # Creates an input-function iterator over a one-element dataset of the inputs, initializes it,
    # runs comm_fn on the strategy and checks each output equals the expected value.
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensors(inputs))

    self.evaluate(inputs.initialize())
    outputs = self.evaluate(
        list(
            map(strategy.experimental_local_results,
                strategy.experimental_run(comm_fn, inputs))))
    self.assertAllEqual([expected[0]], outputs[0])
    self.assertAllEqual([expected[1]], outputs[1])

  def _test_collective_comms_gradients(self, strategy, comm_fn, inputs,
                                       expected_grads):
    # [FUNCTION DOC]
    # HELPER: shared test body for graph-mode gradients through a collective. Inputs: strategy,
    # comm_fn, inputs (list of multipliers) and expected_grads. Skipped in eager execution.
    # KNOWN ISSUE: same late-binding of the name inputs (rebound to the iterator) as the sibling
    # helpers; works only because the dataset lambda is called first.
    # Builds a step that applies comm_fn to a constant and multiplies by the input, returning the
    # gradient with respect to the constant; runs it on the strategy and compares local results
    # with expected_grads.
    if context.executing_eagerly():
      self.skipTest("`tf.gradients` is not supported with eager execution.")

    def step(c):
      # [FUNCTION DOC]
      # Per-replica step: applies the collective callback to a constant and multiplies by the
      # input c; returns the gradient of that product with respect to the constant (computed with
      # graph gradients or a gradient tape, depending on the enclosing test).
      x = array_ops.identity(42.)
      y = comm_fn(x) * c
      return gradients_impl.gradients(y, [x])[0]

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensors(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(step, inputs))))

  def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs,
                                           expected_grads):
    # [FUNCTION DOC]
    # HELPER: shared test body for tape gradients through a collective. Inputs: strategy,
    # comm_fn, inputs (list of multipliers) and expected_grads.
    # KNOWN ISSUE: same late-binding of the name inputs (rebound to the iterator) as the sibling
    # helpers; works only because the dataset lambda is called first.
    # Builds a step that watches a constant on a gradient tape, applies comm_fn, multiplies by the
    # input and returns the tape gradient; runs it on the strategy and compares local results
    # with expected_grads.

    def step(c):
      # [FUNCTION DOC]
      # Per-replica step: applies the collective callback to a constant and multiplies by the
      # input c; returns the gradient of that product with respect to the constant (computed with
      # graph gradients or a gradient tape, depending on the enclosing test).
      x = array_ops.identity(42.)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = comm_fn(x) * c
      return tape.gradient(y, x)

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensors(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(step, inputs))))

  def _test_device_and_input_device_are_colocated(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body, skipped in eager execution. Starts a local two-worker cluster,
    # builds an input iterator over a range of five and a run op adding one, then opens a session
    # against the second worker, initializes the iterator and runs the op. Passes if it runs
    # without error (the input device is colocated with the computation device). Side effect:
    # starts in-process servers.
    if context.executing_eagerly():
      self.skipTest(
          "cross-device tests are not supported with eager execution.")
    workers, _ = test_util.create_local_cluster(2, 0)
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.range(5))
    comm_fn = lambda x: x + 1
    run_op = strategy.experimental_run(comm_fn, inputs)
    with session_lib.Session(target=workers[1].target) as sess:
      sess.run(inputs.initialize())
      sess.run(run_op)

  def _test_device_and_input_device_are_colocated_with_function(self, strategy):
    # [FUNCTION DOC]
    # HELPER: shared test body, skipped in eager execution. Same as the colocation test but the
    # strategy run is wrapped as a traced function and called inside a device scope for the second
    # worker's CPU, so the function itself is defined on the right device. Runs the op in a
    # session on the second worker. Side effect: starts in-process servers.
    if context.executing_eagerly():
      self.skipTest(
          "cross-device tests are not supported with eager execution.")
    workers, _ = test_util.create_local_cluster(2, 0)
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.range(5))
    comm_fn = lambda x: x + 1
    experimental_run = def_function.function()(strategy.experimental_run)
    with ops.device("/job:worker/replica:0/task:1/device:CPU:0"):
      # The tf.function must be defined on the right device as well.
      run_op = experimental_run(comm_fn, inputs)
    with session_lib.Session(target=workers[1].target) as sess:
      sess.run(inputs.initialize())
      sess.run(run_op)


class TwoDeviceDistributionTestBase(test.TestCase):
  # [CLASS DOC]
  # Base test case with reusable test bodies for strategies that have exactly two devices (two
  # replicas): run semantics, all-reduce sum and mean (values, graph gradients, tape gradients),
  # each optionally executed inside a traced function. Concrete tests call the underscore
  # methods with a strategy.
  """Some tests that should work with any two-device DistributionStrategy."""

  def _test_run(self, strategy, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body. Input: strategy and run_in_function (boolean, default false; run
    # callbacks as traced functions). Runs a function returning replica id plus one (expects one
    # and two); then a function giving double and square of that (expects two and four, and one
    # and four); then a keyword-argument function a plus two times b plus two (expects six and
    # fourteen).
    out1 = strategy.run(_maybe_run_in_function(
        lambda: distribute_lib.get_replica_context().replica_id_in_sync_group + 1,
        run_in_function))
    self.assertAllEqual([1, 2], self.evaluate(strategy.unwrap(out1)))

    out2 = strategy.run(_maybe_run_in_function(
        lambda x: {"a": x * 2, "b": x * x}, run_in_function), args=(out1,))
    out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
    self.assertAllEqual([2, 4], out2_vals["a"])
    self.assertAllEqual([1, 4], out2_vals["b"])

    out3 = strategy.run(_maybe_run_in_function(
        lambda b, a: a + 2 * b + 2, run_in_function), kwargs=out2)
    self.assertAllEqual([6, 14], self.evaluate(strategy.unwrap(out3)))

  def _test_all_reduce_sum(self, strategy, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body: all-reduce sum over two replicas with scalar inputs one and
    # three, and vector inputs; expects four and the elementwise sums forty-two, forty-three on
    # both replicas. run_in_function (default false) traces the callbacks.
    self._test_collective_comms(
        strategy,
        _all_sum,
        inputs=([1., 3.], [[39., 2.], [3., 41.]]),
        expected=(4., [42., 43.]),
        run_in_function=run_in_function)

  def _test_all_reduce_sum_gradients(self, strategy, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body: graph gradients of an all-reduce sum with multipliers one and
    # three; each replica must see four. run_in_function (default false) traces the step.
    self._test_collective_comms_gradients(
        strategy, _all_sum, inputs=[1., 3.], expected_grads=[4., 4.],
        run_in_function=run_in_function)

  def _test_all_reduce_sum_gradient_tape(self, strategy, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body: tape gradients of an all-reduce sum with multipliers one and
    # three; each replica must see four. run_in_function (default false) traces the step.
    self._test_collective_comms_gradient_tape(
        strategy, _all_sum, inputs=[1., 3.], expected_grads=[4., 4.],
        run_in_function=run_in_function)

  def _test_all_reduce_mean(self, strategy, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body: all-reduce mean over two replicas with the same inputs as the sum
    # test; expects two and twenty-one, twenty-one-and-a-half on both replicas. run_in_function
    # (default false) traces the callbacks.
    self._test_collective_comms(
        strategy,
        _all_mean,
        inputs=([1., 3.], [[39., 2.], [3., 41.]]),
        expected=(2., [21., 21.5]),
        run_in_function=run_in_function)

  def _test_all_reduce_mean_gradients(self, strategy, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body: graph gradients of an all-reduce mean with multipliers one and
    # three; each replica must see two. run_in_function (default false) traces the step.
    self._test_collective_comms_gradients(
        strategy, _all_mean, inputs=[1., 3.], expected_grads=[2., 2.],
        run_in_function=run_in_function)

  def _test_all_reduce_mean_gradient_tape(self, strategy,
                                          run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body: tape gradients of an all-reduce mean with multipliers one and
    # three; each replica must see two. run_in_function (default false) traces the step.
    self._test_collective_comms_gradient_tape(
        strategy, _all_mean, inputs=[1., 3.], expected_grads=[2., 2.],
        run_in_function=run_in_function)

  def _test_collective_comms(self, strategy, comm_fn, inputs, expected,
                             run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body. Inputs: strategy, comm_fn (collective callback), inputs (pair of
    # per-replica scalars and per-replica vectors), expected (pair of expected scalar and vector)
    # and run_in_function (boolean, default false).
    # KNOWN ISSUE: the name inputs is rebound to the iterator while the dataset-producing lambda
    # still refers to it by name; works only because the lambda is called before rebinding.
    # Builds an input iterator slicing the inputs over the two replicas, runs comm_fn (traced if
    # requested) and checks both replicas received the expected scalar and vector.
    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))

    self.evaluate(inputs.initialize())
    outputs = self.evaluate(
        list(
            map(strategy.experimental_local_results,
                strategy.experimental_run(
                    _maybe_run_in_function(comm_fn, run_in_function), inputs))))
    self.assertAllEqual([expected[0], expected[0]], outputs[0])
    self.assertAllEqual([expected[1], expected[1]], outputs[1])

  def _test_collective_comms_gradients(self, strategy, comm_fn, inputs,
                                       expected_grads, run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body for graph gradients through a collective. Inputs: strategy,
    # comm_fn, inputs (list of per-replica multipliers), expected_grads and run_in_function
    # (boolean, default false). Skipped in eager mode unless run in a traced function.
    # KNOWN ISSUE: same late-binding of the name inputs (rebound to the iterator) as the sibling
    # helpers; works only because the dataset lambda is called first.
    # Runs a gradient-computing step per replica and compares local results with expected_grads.
    if context.executing_eagerly() and not run_in_function:
      self.skipTest("`tf.gradients` is not supported with eager execution "
                    "without using tf.functions.")

    def step(c):
      # [FUNCTION DOC]
      # Per-replica step: applies the collective callback to a constant and multiplies by the
      # input c; returns the gradient of that product with respect to the constant (computed with
      # graph gradients or a gradient tape, depending on the enclosing test).
      x = array_ops.identity(42.)
      y = comm_fn(x) * c
      return gradients_impl.gradients(y, [x])[0]

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(
                    _maybe_run_in_function(step, run_in_function), inputs))))

  def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs,
                                           expected_grads,
                                           run_in_function=False):
    # [FUNCTION DOC]
    # HELPER: shared test body for tape gradients through a collective. Inputs: strategy,
    # comm_fn, inputs (list of per-replica multipliers), expected_grads and run_in_function
    # (boolean, default false).
    # KNOWN ISSUE: same late-binding of the name inputs (rebound to the iterator) as the sibling
    # helpers; works only because the dataset lambda is called first.
    # Runs a tape-gradient step per replica and compares local results with expected_grads.

    def step(c):
      # [FUNCTION DOC]
      # Per-replica step: applies the collective callback to a constant and multiplies by the
      # input c; returns the gradient of that product with respect to the constant (computed with
      # graph gradients or a gradient tape, depending on the enclosing test).
      x = array_ops.identity(42.)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = comm_fn(x) * c
      return tape.gradient(y, x)

    inputs = strategy.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))

    self.evaluate(inputs.initialize())
    self.assertAllEqual(
        expected_grads,
        self.evaluate(
            strategy.experimental_local_results(
                strategy.experimental_run(
                    _maybe_run_in_function(step, run_in_function),
                    inputs))))


class RemoteSingleWorkerMirroredStrategyBase(DistributionTestBase):
  # [CLASS DOC]
  # Base test case for strategies connected to a remote single worker. Extends the generic
  # distribution test base with test bodies on replica count, loss minimization, device scope and
  # input-function iterators. Subclasses must override the method reporting the number of GPUs.
  """Tests for a Remote single worker."""

  def _get_num_gpus(self):
    # [FUNCTION DOC]
    # HELPER: abstract hook meant to be overridden by subclasses to return the number of GPUs
    # (replicas) on the remote worker. The base version does nothing and returns nothing/null, so
    # tests using it without overriding fail.
    pass

  def _testNumReplicasInSync(self, distribution):
    # [FUNCTION DOC]
    # HELPER: shared test body. Asserts the strategy's replica count equals the subclass-provided
    # number of GPUs.
    self.assertEqual(self._get_num_gpus(), distribution.num_replicas_in_sync)

  def _testMinimizeLoss(self, distribution):
    # [FUNCTION DOC]
    # HELPER: shared test body. Runs the eager loss-minimization test in eager mode, otherwise
    # the graph-mode test with learning rate 0.05.
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution, learning_rate=0.05)

  def _testDeviceScope(self, distribution):
    # [FUNCTION DOC]
    # HELPER: shared test body. In the strategy scope creates a constant (default placement) and
    # one under a CPU device scope. Asserts the first sits on the remote worker task zero (with CPU
    # device suffix when eager, no device suffix in graph mode) and the second on the worker's
    # CPU device.
    with distribution.scope():
      a = array_ops.identity(1.)
      with ops.device("/cpu:0"):
        b = array_ops.identity(1.)
      if context.executing_eagerly():
        device = "/job:worker/replica:0/task:0/device:CPU:0"
      else:
        device = "/job:worker/replica:0/task:0"
      self.assertEqual(a.device, device)
      self.assertEqual(b.device, "/job:worker/replica:0/task:0/device:CPU:0")

  def _testMakeInputFnIteratorWithDataset(self, distribution):
    # [FUNCTION DOC]
    # HELPER: shared test body. Uses an input function returning a range-of-one-hundred dataset,
    # one worker and the subclass GPU count. Expected values are consecutive groups of one number
    # per replica. The input function is wrapped with the context-checking helper (pipeline id
    # equal to call counter). Creates the strategy's input-function iterator and runs the
    # iterator check helper in a cached session.
    dataset_fn = lambda: dataset_ops.Dataset.range(100)
    num_gpus = self._get_num_gpus()  # pylint: disable=assignment-from-no-return
    num_workers = 1

    expected_values = [[i+j for j in range(num_gpus)] * num_workers
                       for i in range(0, 100, num_gpus)]

    # Dummy cached_session is used in Eager
    with self.cached_session() as sess:
      # `expected_input_pipeline_id` is None because the input_fn will be called
      # multiple times, each with a different input_pipeline_id.
      input_fn = self._input_fn_to_test_input_context(
          dataset_fn,
          expected_num_replicas_in_sync=num_workers*num_gpus,
          expected_num_input_pipelines=num_workers,
          expected_input_pipeline_id=None)
      iterator = distribution.make_input_fn_iterator(input_fn)
      self._test_input_fn_iterator(
          iterator, distribution.extended.worker_devices, expected_values, sess)

  def _testMakeInputFnIteratorWithCallable(self, distribution):
    # [FUNCTION DOC]
    # HELPER: shared test body. Like the dataset variant but the input function returns the
    # next-element callable of a one-shot iterator over a range of one hundred. Runs the iterator
    # check without reinitialization and with order-insensitive comparison.
    def fn():
      # [FUNCTION DOC]
      # Input-producing function: builds a range-of-one-hundred dataset, a one-shot iterator over
      # it, and returns the iterator's next-element callable (not an element).
      dataset = dataset_ops.Dataset.range(100)
      it = dataset_ops.make_one_shot_iterator(dataset)
      return it.get_next

    num_gpus = self._get_num_gpus()  # pylint: disable=assignment-from-no-return
    num_workers = 1

    expected_values = []
    for i in range(0, 100, num_gpus):
      expected_values.append([i+j for j in range(num_gpus)] * num_workers)

    # Dummy cached_session is used in Eager
    with self.cached_session() as sess:
      # `expected_input_pipeline_id` is None because the input_fn will be called
      # multiple times, each with a different input_pipeline_id.
      input_fn = self._input_fn_to_test_input_context(
          fn,
          expected_num_replicas_in_sync=num_workers*num_gpus,
          expected_num_input_pipelines=num_workers,
          expected_input_pipeline_id=None)
      iterator = distribution.make_input_fn_iterator(input_fn)
      self._test_input_fn_iterator(
          iterator, distribution.extended.worker_devices, expected_values, sess,
          test_reinitialize=False, ignore_order=True)


def _all_sum(value):
  # [FUNCTION DOC]
  # HELPER: collective test callback. Input: value (tensor or structure). Gets the current
  # replica context and returns the sum all-reduce of value across replicas.
  ctx = distribute_lib.get_replica_context()
  return ctx.all_reduce(reduce_util.ReduceOp.SUM, value)


def _all_mean(value):
  # [FUNCTION DOC]
  # HELPER: collective test callback. Input: value. Gets the current replica context and
  # returns the mean all-reduce of value across replicas.
  ctx = distribute_lib.get_replica_context()
  return ctx.all_reduce(reduce_util.ReduceOp.MEAN, value)
