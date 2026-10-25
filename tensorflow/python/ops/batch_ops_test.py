# [MODULE DOC]
# Tests for the in-graph batching operations and the batch function decorator: the low-level batch
# and unbatch operations, the batch function operation (with captured inputs, errors in inputs,
# large-batch splitting), the decorator wrapper, device placement behavior (GPU, soft placement),
# unbatch timeouts, and the invalid-argument handling of the unbatch and unbatch-gradient kernels.
# Most tests are graph-mode only (they return immediately when running eagerly) and use helper
# threads to issue concurrent session runs so that requests are batched together. Defines one
# small helper function and one test class that runs in both graph and eager modes. No import-time
# side effects beyond imports; running as a script starts the standard test runner.
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

"""Tests for the currently experimental in-graph batch ops."""
import threading
import time
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import batch_ops
from tensorflow.python.ops import gen_batch_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def delayed_plus1(x):
  # [FUNCTION DOC]
  # HELPER: slow computation injected through a Python-function operation by the unbatch timeout
  # test to delay one pipeline. Waits a tenth of a second, then returns the input plus one.
  # Input: a numeric value or array. Output: the incremented value. Side effect: sleeps.
  """Sleeps for 100ms then returns x+1."""
  time.sleep(0.1)
  return x + 1


@test_util.run_all_in_graph_and_eager_modes
class BatchOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the batch, unbatch, batch function operations and the batch function decorator.
  # Decorated to run every test in both graph and eager modes, though most tests return early in
  # eager mode because batching in an eager context is not yet supported. Tests use worker
  # threads to issue concurrent session runs against shared queues. No shared fixtures.
  """Tests for batch_ops.{un,}batch."""

  # Test for only non eager mode as batching in eager context as a functionality
  # is TBD.
  def testBasicBatch(self):
    # [FUNCTION DOC]
    # Test (graph mode only): creates a batch operation with batch size 2 and a very long timeout
    # over a one-element integer placeholder. One run happens in a worker thread and one in the
    # main thread. Asserts that exactly one of them received the full batch containing both
    # values (1 and 2) with a two-row index tensor, and that the other received empty outputs.
    """Tests that a single batched tensor executes together and only once."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, index, _ = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=2,
          batch_timeout_micros=36000000, grad_timeout_micros=0,
          batching_queue="")
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the batched and index outputs with the
        # input fed as a single element one and appends the fetched results to the shared results
        # list. Returns nothing.
        thread_results.extend(
            sess.run([batched, index], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([batched, index], feed_dict={inp: [2]})
      worker_thread.join()

      # At this point either the thread or the main did the batch and the other
      # should have empty results.
      if list(thread_results[0][0]):
        batch_t = thread_results[0][0]
        index_t = thread_results[1]
        empty_b = main_results[0][0]
        empty_m = main_results[1]
      else:
        batch_t = main_results[0][0]
        index_t = main_results[1]
        empty_b = thread_results[0][0]
        empty_m = thread_results[1]

      # Check that both the inputs made it out exactly once.
      self.assertAllEqual(sorted(batch_t), (1, 2))
      # Check that we get 2 rows in the index tensor.
      self.assertEqual(len(index_t), 2)
      # Check that the other ones are empty.
      self.assertEqual(len(empty_b), 0)
      self.assertEqual(len(empty_m), 0)

  def testBatchWithPadding(self):
    # [FUNCTION DOC]
    # Test (graph mode only): batch operation with maximum size 10 and allowed sizes 5 and 10,
    # fed two requests of two elements each from two threads. Asserts that the produced batch is
    # padded up to length 5, the smallest allowed size that fits the four real elements.
    """Test that batching with padding up to an allowed batch size works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[2])
      batched, index, _ = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[5, 10],
          grad_timeout_micros=0, batching_queue="")
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the batched and index outputs with the
        # input fed as the two elements one and three and appends the results to the shared list.
        # Returns nothing.
        thread_results.extend(
            sess.run([batched, index], feed_dict={inp: [1, 3]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([batched, index], feed_dict={inp: [2, 4]})
      worker_thread.join()

      # At this point either the thread or the main did the batch and the other
      # should have empty results.
      if list(thread_results[0][0]):
        batch_t = thread_results[0][0]
      else:
        batch_t = main_results[0][0]

      # Check that the batch tensor incorporates the padding.
      self.assertEqual(len(batch_t), 5)

  def testMultipleBatch(self):
    # [FUNCTION DOC]
    # Test (graph mode only): batches two input tensors together in one batch operation (size 2,
    # long timeout), fed from a worker thread and the main thread. Asserts that one caller gets
    # both tensors batched (values 1,2 for the first and 2,3 for the second, in some order) and
    # the other caller gets empty tensors for both.
    """Tests that multiple batched tensors execute together."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp0 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      inp1 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, _, _ = batch_ops.batch(
          [inp0, inp1],
          num_batch_threads=1,
          max_batch_size=2,
          batch_timeout_micros=36000000,
          grad_timeout_micros=0,
          batching_queue="")
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the batched outputs with the first
        # input fed one and the second fed two, appending results to the shared list. Returns
        # nothing.
        thread_results.extend(
            sess.run([batched], feed_dict={inp0: [1],
                                           inp1: [2]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([batched], feed_dict={inp0: [2], inp1: [3]})
      worker_thread.join()

      # At this point either the thread or the main did the batch and the other
      # should have empty results.
      if list(thread_results[0][0]):
        batch_t = thread_results[0]
        empty_t = main_results[0]
      else:
        batch_t = main_results[0]
        empty_t = thread_results[0]

      # Assert that the tensors were batched together.
      self.assertAllEqual(sorted(batch_t[0]), [1, 2])
      self.assertAllEqual(sorted(batch_t[1]), [2, 3])
      self.assertAllEqual(empty_t[0], [])
      self.assertAllEqual(empty_t[1], [])

  def testIllegalBatchDifferentDim0Sizes(self):
    # [FUNCTION DOC]
    # Test (graph mode only): feeds the batch operation two tensors whose first dimensions differ
    # (1 and 2) with zero timeout, and asserts that running it raises an error whose message
    # contains the phrase saying the tensors must have equal 0th-dimension size.
    """Tests illegally feeding tensors with different dim0 sizes."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp0 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      inp1 = array_ops.placeholder(dtype=dtypes.int32, shape=[2])
      batched, index, _ = batch_ops.batch(
          [inp0, inp1], num_batch_threads=1, max_batch_size=2,
          batch_timeout_micros=0, grad_timeout_micros=0, batching_queue="")
      with self.assertRaises(Exception) as raised:
        _ = sess.run([batched, index], feed_dict={inp0: [0], inp1: [1, 2]})
      self.assertGreater(
          raised.exception.message.find("must have equal 0th-dimension size"),
          0)

  def testBasicUnbatch(self):
    # [FUNCTION DOC]
    # Test (graph mode only): batches an input (allowed sizes 3 and 10), adds one to the batched
    # tensor, and unbatches the result with the index and id tensors. Two concurrent callers feed
    # values 1 and 2 and each must get back its own value plus one (2 and 3 respectively).
    """Tests that batch and unbatch work together."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, index, id_t = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[3, 10],
          grad_timeout_micros=0, batching_queue="")
      computation = batched[0] + 1
      result = batch_ops.unbatch(computation, index, id_t,
                                 timeout_micros=1000000, shared_name="unbatch")
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the unbatched result with the input fed
        # as one and appends the fetched value to the shared list. Returns nothing.
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBasicUnbatchDecorated(self):
    # [FUNCTION DOC]
    # Test (graph mode only): applies the batch function decorator (one thread, size 10, long
    # timeout) to a function adding one, which also asserts its input has a known shape object.
    # Two concurrent callers feed values 1 and 2 and must receive 2 and 3. A dummy placeholder
    # with default is created first because removing it makes the test flaky (known open to-do).
    """Tests that the batch_function decorator works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      # TODO(apassos): Removing this line causes test flakiness! Ideally should
      # be investigated.
      default_inp = array_ops.placeholder_with_default(2, shape=[])  # pylint: disable=unused-variable

      @batch_ops.batch_function(1, 10, 100000)
      def computation(in_t):
        # [FUNCTION DOC]
        # HELPER: the function being batched in the enclosing test. Asserts the input tensor has
        # a shape object and returns the input plus one.
        self.assertTrue(in_t.shape is not None)
        return in_t + 1

      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      result = computation(inp)
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the decorated function's result with
        # the input fed as one and appends the value to the shared list. Returns nothing.
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testUnbatchInvalidIdArg(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring ("unbatch work together") is a copy of another test's text and
    # does not describe the test; it only runs in eager mode and silently does nothing (passes)
    # in graph mode.
    # Test (eager mode only): calls the unbatch operation with a non-scalar id tensor (random
    # 3x3x1 integers) together with random batched tensor and batch index, and asserts it raises
    # an invalid-argument error whose message says the input id should be scalar.
    """Tests that unbatch work together."""
    if context.executing_eagerly():
      batched_tensor = constant_op.constant(
          value=np.random.random(size=(3, 3, 1)), dtype=dtypes.float64)
      batched_index = constant_op.constant(
          value=np.random.randint(0, 100, size=(3, 3, 1)), dtype=dtypes.int64)
      arg_id = constant_op.constant(
          value=np.random.randint(0, 100, size=(3, 3, 1)), dtype=dtypes.int64)

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Input id should be scalar;"):
        batch_ops.unbatch(
            batched_tensor=batched_tensor,
            batch_index=batched_index,
            id=arg_id,
            timeout_micros=50,
            container="",
            shared_name="")

  def testBatchDecoratedWithCapturedInput(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is identical to the plain decorator test and does not mention the
    # captured inputs that this test is actually about.
    # Test (graph mode only): decorates a function that adds three captured values (a placeholder
    # with default 2, a variable 3 on the default device and a variable 4 pinned to the CPU) to
    # its input. After initializing variables, two concurrent callers feed 1 and 2 and must get 10
    # and 11.
    """Tests that the batch_function decorator works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      captured_inp0 = array_ops.placeholder_with_default(2., shape=[])
      captured_inp1 = resource_variable_ops.ResourceVariable(3.)
      with ops.device("/cpu:0"):
        captured_inp2 = resource_variable_ops.ResourceVariable(4.)

      @batch_ops.batch_function(1, 10, 100000)
      def computation(in_t):
        # [FUNCTION DOC]
        # HELPER: the function being batched; returns the input plus the three captured values
        # of the enclosing test.
        return in_t + captured_inp0 + captured_inp1 + captured_inp2

      inp = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
      result = computation(inp)
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the decorated result with the input fed
        # as one and appends the value to the shared list. Returns nothing.
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      sess.run(variables.global_variables_initializer())
      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [10])
      self.assertEqual(main_results[0], [11])

  @test_util.disable_xla("DeviceIndex returns sentinel value with XLA")
  def testBatchDecoratedGpu(self):
    # [FUNCTION DOC]
    # Test (graph mode only, skipped under XLA): decorates a function that adds the device index
    # (0 on CPU, 1 on GPU) to its input. Two concurrent callers feed 10 and 20 and must get those
    # values plus one if a GPU is available, otherwise plus zero, showing the batched function
    # runs on the GPU when present.
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:

      @batch_ops.batch_function(1, 10, 100000)
      def computation(in_t):
        # [FUNCTION DOC]
        # HELPER: the function being batched; queries the device index operation (zero on CPU,
        # one on GPU) and returns the input plus that index as a float.
        # index is 0 on CPU and 1 on GPU
        index = gen_functional_ops.DeviceIndex(device_names=["CPU", "GPU"])
        return in_t + math_ops.cast(index, dtypes.float32)

      inp = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
      result = computation(inp)
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the result with the input fed as the
        # float ten and appends the value to the shared list. Returns nothing.
        thread_results.extend(sess.run([result], feed_dict={inp: [10.]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [20.]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [10 + test_util.is_gpu_available()])
      self.assertEqual(main_results[0], [20 + test_util.is_gpu_available()])

  def testParallelRunsWithCpuAndGpu(self):
    # [FUNCTION DOC]
    # Test (graph mode only): regression test for a bug where send nodes of one batch function
    # call delivered tensors to receive nodes of a different call. Decorates a function that adds
    # one on the GPU then one on the CPU, calls it ten times with distinct placeholders in one
    # graph and runs all results in a single session run, asserting each output equals its input
    # plus two.
    # Run multiple instances of a batch function in parallel. This is a
    # regression test: this used to fail because _Send nodes for one call would
    # send the tensor to the _Recv node for a different call.
    if context.executing_eagerly():
      return
    @batch_ops.batch_function(1, 2, 1)
    def f(x):
      # [FUNCTION DOC]
      # HELPER: the function being batched; adds one to the input under a GPU device scope, then
      # adds one more under a CPU device scope and returns the result.
      with ops.device("/GPU:0"):
        x = x + 1.
      with ops.device("/CPU:0"):
        return x + 1
    num_calls = 10
    placeholders = [array_ops.placeholder(dtypes.float32, shape=(1,))
                    for _ in range(num_calls)]
    results = []
    for p in placeholders:
      result = f(p)
      results.append(result)
    inputs = [[float(i)] for i in range(num_calls)]
    expected = [[float(i + 2)] for i in range(num_calls)]
    with self.session() as sess:
      outputs = sess.run(results, feed_dict=dict(zip(placeholders, inputs)))
      self.assertAllEqual(outputs, expected)

  def testSoftPlacement(self):
    # [FUNCTION DOC]
    # Test (graph mode only): decorates a function that explicitly places an addition on the GPU.
    # With soft placement allowed the session run succeeds even without a GPU. With soft placement
    # disabled, it succeeds if a GPU exists, otherwise it must raise an invalid-argument error
    # saying a device cannot be assigned for the operation.
    if context.executing_eagerly():
      return

    @batch_ops.batch_function(1, 10, 100000)
    def computation(in_t):
      # [FUNCTION DOC]
      # HELPER: the function being batched; adds one to the input under an explicit GPU device
      # scope and returns it.
      with ops.device("/GPU:0"):
        return in_t + 1.

    inp = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
    result = computation(inp)

    # With soft placement, the function will run even without a GPU
    config = config_pb2.ConfigProto(allow_soft_placement=True)
    with self.session(config=config) as sess:
      sess.run([result], feed_dict={inp: [20.]})

    # Without soft placement, the function fails without a GPU due to the
    # addition explicitly being placed on the GPU
    config.allow_soft_placement = False
    with self.session(config=config) as sess:
      if test_util.is_gpu_available():
        sess.run([result], feed_dict={inp: [20.]})
      else:
        with self.assertRaisesRegex(InvalidArgumentError,
                                    "Cannot assign a device for operation"):
          sess.run([result], feed_dict={inp: [20.]})

  def testBatchFunctionOp(self):
    # [FUNCTION DOC]
    # Test (graph mode only): calls the generated batch function operation directly with a
    # graph-defined function adding one, with no extra captured inputs. Two concurrent callers
    # feed 1 and 2 and must get 2 and 3.
    """Tests that the batch_function op works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:

      @function.Defun(dtypes.int32)
      def computation(in_t):
        # [FUNCTION DOC]
        # HELPER: graph-defined function passed to the batch function operation; returns its
        # integer input plus one.
        return in_t + 1

      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      result = gen_batch_ops.batch_function(
          [inp],
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,
          Tout=[dtypes.int32],
          f=computation,
          captured_tensors=computation.captured_inputs)
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the operation result with the input fed
        # as one and appends the value to the shared list. Returns nothing.
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBatchFunctionOpWithCapturedInput(self):
    # [FUNCTION DOC]
    # Test (graph mode only): calls the batch function operation with a graph-defined function
    # that adds one captured value and subtracts another (placeholders with defaults 2 and 1),
    # passing the captured tensors explicitly and allowed sizes 3 and 10. Two concurrent callers
    # feed 1 and 2 and must get 2 and 3.
    """Tests that batch_function op works with captured input."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      captured_inp0 = array_ops.placeholder_with_default(2, shape=[])
      captured_inp1 = array_ops.placeholder_with_default(1, shape=[])
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])

      @function.Defun(dtypes.int32)
      def computation(inp):
        # [FUNCTION DOC]
        # HELPER: graph-defined function passed to the batch function operation; returns its input
        # plus the first captured value minus the second.
        return inp + captured_inp0 - captured_inp1

      result = gen_batch_ops.batch_function(
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[3, 10],
          batching_queue="",
          f=computation,
          in_tensors=[inp],
          captured_tensors=computation.captured_inputs,
          Tout=[o.type for o in computation.definition.signature.output_arg])

      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the operation result with the input fed
        # as one and appends the value to the shared list. Returns nothing.
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [2]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testBatchFunctionOpWithInputError(self):
    # [FUNCTION DOC]
    # Test (graph mode only): calls the batch function operation with a function expecting two
    # inputs but supplies only one, and asserts that running it raises an invalid-argument error
    # saying the function takes 2 arguments but 1 was passed.
    """Tests that batch_function op works with error in the inputs."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])

      @function.Defun(dtypes.int32, dtypes.int32)
      def computation(in0, in1):
        # [FUNCTION DOC]
        # HELPER: graph-defined two-input function used to provoke an argument count mismatch;
        # returns the sum of its two inputs.
        return in0 + in1

      result = gen_batch_ops.batch_function(
          [inp],  # computation actually expects 2 inputs.
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          batching_queue="",
          f=computation,
          captured_tensors=computation.captured_inputs,
          Tout=[o.type for o in computation.definition.signature.output_arg])

      with self.assertRaisesRegex(
          InvalidArgumentError,
          r"Function takes 2 argument\(s\) but 1 argument\(s\) were passed"):
        sess.run([result], feed_dict={inp: [2]})

  def testBatchFunctionOpWithLargeBatchSplitted(self):
    # [FUNCTION DOC]
    # Test (graph mode only): calls the batch function operation with large batch splitting on,
    # maximum size 5 and allowed sizes 1 and 2, using a function adding three. Three concurrent
    # callers feed 5 elements, 1 element and 3 elements; the large inputs get split and later
    # recombined. Asserts each caller receives its own inputs plus three, in the original order.
    """Tests that the batch_function op works with large batch splitted."""
    if context.executing_eagerly():
      return

    with self.cached_session() as sess:

      @function.Defun(dtypes.int32)
      def computation(in_t):
        # [FUNCTION DOC]
        # HELPER: graph-defined function passed to the batch function operation; returns its
        # integer input plus three.
        return in_t + 3

      inp = array_ops.placeholder(dtype=dtypes.int32)
      result = gen_batch_ops.batch_function(
          [inp],
          num_batch_threads=2,
          # enable_large_batch_splitting is True, so it's valid as long as
          # max('allowed_batch_sizes') <= 'max_batch_size'.
          allowed_batch_sizes=[1, 2],
          max_batch_size=5,
          batch_timeout_micros=100000,  # 100ms
          Tout=[dtypes.int32],
          enable_large_batch_splitting=True,
          f=computation,
          captured_tensors=computation.captured_inputs)
      thread1_results = []
      thread2_results = []

      # Input sizes of worker1 and main thread are larger than
      # max(allowed_batch_sizes), while input size of worker2 is smaller.
      def worker1():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the result with a five-element input
        # (5 through 9) and appends the fetched value to the first thread's results list.
        # Returns nothing.
        thread1_results.extend(
            sess.run([result], feed_dict={inp: [5, 6, 7, 8, 9]}))

      worker_thread1 = threading.Thread(target=worker1)
      worker_thread1.start()

      def worker2():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the result with a one-element input
        # (10) and appends the fetched value to the second thread's results list. Returns nothing.
        thread2_results.extend(sess.run([result], feed_dict={inp: [10]}))

      worker_thread2 = threading.Thread(target=worker2)
      worker_thread2.start()

      main_results = sess.run([result], feed_dict={inp: [2, 3, 4]})
      worker_thread1.join()
      worker_thread2.join()
      self.assertTrue(
          np.all(np.equal(thread2_results[0], np.array([13], dtype=np.int32))))
      self.assertTrue(
          np.all(
              np.equal(thread1_results[0],
                       np.array([8, 9, 10, 11, 12], dtype=np.int32))))
      self.assertTrue(
          np.all(
              np.equal(main_results[0], np.array([5, 6, 7], dtype=np.int32))))

  def testBasicUnbatchDecoratedWithReshape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is identical to the plain decorator test and does not mention the
    # reshape that this test exercises.
    # Test (graph mode only): decorates a function that flattens a 1x1 input to a vector and adds
    # one. Two concurrent callers feed [[1]] and [[2]] and must get back 2 and 3 as vectors.
    """Tests that the batch_function decorator works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:

      @batch_ops.batch_function(1, 10, 100000)
      def computation(in_t):
        # [FUNCTION DOC]
        # HELPER: the function being batched; flattens its input to one dimension and adds one.
        return array_ops.reshape(in_t, [-1]) + 1

      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1, 1])
      result = computation(inp)
      thread_results = []

      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the result with the input fed as a
        # one-by-one matrix holding one and appends the value to the shared list. Returns nothing.
        thread_results.extend(sess.run([result], feed_dict={inp: [[1]]}))

      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      main_results = sess.run([result], feed_dict={inp: [[2]]})
      worker_thread.join()
      self.assertEqual(thread_results[0], [2])
      self.assertEqual(main_results[0], [3])

  def testUnbatchTimeout(self):
    # [FUNCTION DOC]
    # Test (graph mode only): builds two pipelines sharing one unbatch resource with a 10
    # microsecond timeout: a normal one and one whose computation is delayed by a Python-function
    # operation. A worker thread starts the normal pipeline first, then the main thread runs the
    # delayed one so the batched tensor reaches the unbatch late. Asserts the worker's call timed
    # out and returned no results. Relies on the order in which the batch operation emits the
    # empty and the real tensors (noted as a to-do).
    """Tests that the unbatch timeout works."""
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      batched, index, id_t = batch_ops.batch(
          [inp], num_batch_threads=1, max_batch_size=2,
          batch_timeout_micros=36000000, grad_timeout_micros=0,
          batching_queue="")
      computation = batched[0] + 1
      timeout_micros = 10
      result = batch_ops.unbatch(computation, index, id_t, timeout_micros,
                                 shared_name="shared_unbatch")
      # Set up a parallel pipeline that delays the computation, but uses the
      # same unbatch resource object as the non-delayed pipeline.
      computation_delayed = script_ops.py_func(delayed_plus1,
                                               [batched[0]],
                                               dtypes.int32)
      result_delayed = batch_ops.unbatch(computation_delayed,
                                         index,
                                         id_t,
                                         timeout_micros,
                                         shared_name="shared_unbatch")

      thread_results = []
      def worker():
        # [FUNCTION DOC]
        # HELPER: thread body for the enclosing test. Runs the non-delayed pipeline with the input
        # fed as one and appends any fetched value to the shared list (stays empty on timeout).
        # Returns nothing.
        # A first call using the non-delayed pipeline. The batcher will send an
        # empty tensor along the non-delayed pipeline.
        thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
      worker_thread = threading.Thread(target=worker)
      worker_thread.start()
      time.sleep(0.1)  # Ensure the thread's call starts first.
      # A second call using the delayed pipeline.  The batcher will send the
      # batched tensor along the delayed pipeline, thus delaying the arrival of
      # the batched tensor at the unbatch op, relative to the empty tensor.
      #
      # TODO(olston, apassos): Avoid relying on the order in which the batch op
      # emits the empty tensor versus the batched one.
      _ = sess.run([result_delayed], feed_dict={inp: [2]})
      worker_thread.join()
      # The thread's call should hit the timeout, and thus get 0 results.
      self.assertEqual(len(thread_results), 0)

  def testUnbatchGradInvalidId(self):
    # [FUNCTION DOC]
    # Test: calls the unbatch gradient kernel with a batch index with three columns, a two-element
    # id tensor and one-element input and gradient, and asserts it raises an invalid-argument
    # error (the id has the wrong shape).
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(
          gen_batch_ops.unbatch_grad(
              original_input=constant_op.constant([1]),
              batch_index=constant_op.constant([
                  [0, 0, 0],
              ], dtype=dtypes.int64),
              grad=constant_op.constant([
                  1,
              ]),
              id=constant_op.constant([
                  1,
                  1,
              ], dtype=dtypes.int64)))

  def testUnbatchGradInvalidBatchId(self):
    # [FUNCTION DOC]
    # Test: calls the unbatch gradient kernel with a batch index having only two columns (instead
    # of three) and one-element input, gradient and id, and asserts it raises an invalid-argument
    # error.
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(
          gen_batch_ops.unbatch_grad(
              original_input=constant_op.constant([1]),
              batch_index=constant_op.constant([
                  [0, 0],
              ], dtype=dtypes.int64),
              grad=constant_op.constant([
                  1,
              ]),
              id=constant_op.constant([
                  1,
              ], dtype=dtypes.int64)))

  def testUnbatchGradInvalidArgs(self):
    # [FUNCTION DOC]
    # Test: feeds the unbatch gradient kernel randomly generated 3x1 tensors (floats for input and
    # gradient, large random integers for batch index and id) with empty container and shared
    # name, and asserts it raises an invalid-argument error for the malformed arguments.
    original_input = random_ops.random_uniform(
        shape=(3, 1), dtype=dtypes.float64, maxval=None)
    batch_index = random_ops.random_uniform(
        shape=(3, 1), dtype=dtypes.int64, maxval=65536)
    grad = random_ops.random_uniform(
        shape=(3, 1), dtype=dtypes.float64, maxval=None)
    batch_id = random_ops.random_uniform(
        shape=(3, 1), dtype=dtypes.int64, maxval=65536)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(
          gen_batch_ops.unbatch_grad(
              original_input=original_input,
              batch_index=batch_index,
              grad=grad,
              id=batch_id,
              container="",
              shared_name="",
              name=""))

if __name__ == "__main__":
  test.main()
