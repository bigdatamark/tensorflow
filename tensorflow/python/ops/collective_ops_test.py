# [MODULE DOC]
# CPU tests for TensorFlow collective operations: all-reduce (sum, max, min, scalar, half
# precision, timeouts, inside while loops, with scoped-allocator rewrites), broadcast send and
# receive, and all-gather (including shape mismatch errors and unknown-shape inputs). It defines
# one test case class that uses graph-mode sessions with several logical CPU devices as the
# collective group, plus a few tracing-based tests. Importing has no side effects beyond
# imports; running it as a script starts the TensorFlow test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Collective Operations."""

import time

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class CollectiveOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the collective ops module on CPU devices. Holds no instance state; shared
  # helpers (names starting with an underscore) build a session with a given number of logical
  # CPUs, run a collective across them and compare results with expected values. The setup
  # resets the global eager context before each test.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture: resets the global eager context (discarding collective state and
    # device configuration from earlier tests), then runs the base class setup. Returns nothing.
    context._reset_context()  # pylint: disable=protected-access
    super(CollectiveOpTest, self).setUp()

  def _testCollectiveReduce(self,
                            inputs,
                            expected,
                            set_graph_key,
                            communication_hint='auto',
                            fp16=False,
                            instance_key=1,
                            merge_op='Add',
                            final_op='Div',
                            timeout=0,
                            reported_group_size=None):
    # [FUNCTION DOC]
    # HELPER: shared driver for the all-reduce tests.
    # Inputs: inputs (list with one value per participating CPU device), expected (value every
    # device must end up with), set_graph_key (whether to set a collective graph key in the run
    # options), communication_hint (default automatic), fp16 (use half precision instead of
    # single), instance_key (default one), merge_op (default add), final_op (default divide),
    # timeout (seconds, default zero meaning none), reported_group_size (group size passed to
    # the op; defaults to the number of inputs, can be made larger to provoke a timeout).
    # Builds a graph-mode session with one CPU device per input, creates one constant and
    # all-reduce per device, runs them and asserts every result is close to the expected value
    # (looser tolerance for half precision). Returns nothing; failures raise from the run or
    # assertions. Logs each result.
    group_key = 1
    group_size = len(inputs)
    if reported_group_size is None:
      reported_group_size = group_size
    device_type = 'CPU'
    config = config_pb2.ConfigProto(device_count={device_type: group_size})
    devices = ['/{}:{}'.format(device_type, i) for i in range(group_size)]

    with self.session(config=config) as sess:
      colred = []
      for i in range(group_size):
        with ops.device(devices[i]):
          tensor = constant_op.constant(inputs[i], dtype=(
              dtypes.float16 if fp16 else dtypes.float32))
          colred.append(
              collective_ops.all_reduce(
                  tensor,
                  reported_group_size,
                  group_key,
                  instance_key,
                  merge_op,
                  final_op,
                  communication_hint=communication_hint,
                  timeout=timeout))
      run_options = config_pb2.RunOptions()
      if set_graph_key:
        run_options.experimental.collective_graph_key = 1
      results = sess.run(colred, options=run_options)
    tolerance = 1e-3 if fp16 else 1e-5
    for i in range(group_size):
      logging.info('i {} result {} expected {}'.format(i, results[i], expected))
      self.assertAllClose(results[i], expected, rtol=tolerance, atol=tolerance)

  def _testMultipleConcurrentCollectiveReduce(self, t0, t1, expected):
    # [FUNCTION DOC]
    # HELPER: shared driver for the concurrent all-reduce test.
    # Inputs: t0 and t1 (the values on CPU 0 and CPU 1) and expected (the value all results
    # must match). Runs two all-reduce instances (add, then divide) per device in the same
    # group inside one session with deterministic sequential execution enabled, and asserts
    # that all four results are close to the expected value. Returns nothing.
    group_key = 1
    group_size = 2
    num_instances = 2
    all_reduces = []
    config = config_pb2.ConfigProto(device_count={'CPU': group_size})
    config.experimental.collective_deterministic_sequential_execution = True
    with self.session(config=config) as sess:
      for cpu in range(group_size):
        with ops.device('/CPU:%d' % cpu):
          in_tensor = constant_op.constant(t0 if cpu == 0 else t1)
          for instance in range(num_instances):
            all_reduces.append(collective_ops.all_reduce(
                in_tensor, group_size, group_key, instance, 'Add', 'Div'))
      results = sess.run(all_reduces)
    for i in range(group_size * num_instances):
      self.assertAllClose(results[i], expected, rtol=1e-5, atol=1e-5)

  def testCollectiveReduce(self):
    # [FUNCTION DOC]
    # Graph-mode all-reduce of two float vectors with an explicit collective graph key.
    # Expects both devices to get the element-wise average.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(
          inputs=[[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
                  [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]],
          expected=[0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
          set_graph_key=True)

  def testCollectiveAutoGraphKey(self):
    # [FUNCTION DOC]
    # Same all-reduce as the basic test but without setting a collective graph key, so the
    # key is chosen automatically. Expects the element-wise average on both devices.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(
          inputs=[[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
                  [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]],
          expected=[0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
          set_graph_key=False)

  def testFp16Reduce(self):
    # [FUNCTION DOC]
    # All-reduce of half-precision vectors with a graph key; expects the element-wise average
    # within the looser half-precision tolerance.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(
          inputs=[[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
                  [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]],
          expected=[0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
          set_graph_key=True,
          fp16=True)

  def testCollectiveMultipleConcurrentReduce(self):
    # [FUNCTION DOC]
    # Runs two all-reduce instances at once on two CPUs via the concurrent helper and expects
    # all results to equal the element-wise average of the two input vectors.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testMultipleConcurrentCollectiveReduce(
          [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
          [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
          [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2])

  def testCollectiveTimeoutV1(self):
    # [FUNCTION DOC]
    # Graph-mode timeout scenario with three devices and a 4.5 second timeout. First runs a
    # normal all-reduce successfully. Then reports a group size one larger than the number of
    # participants and expects a deadline-exceeded error saying the collective timed out waiting
    # for other workers, and that at least the timeout duration of wall-clock time elapsed.
    timeout = 4.5
    kwargs = dict(
        inputs=[[i + j + 0.1 for i in range(8)] for j in range(3)],
        expected=[1 + i + 0.1 for i in range(8)],
        set_graph_key=True,
        timeout=timeout)

    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(**kwargs)

    start_time = time.time()
    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          errors.DeadlineExceededError,
          'Collective has timed out waiting for other workers'):
        self._testCollectiveReduce(
            reported_group_size=len(kwargs['inputs']) + 1, **kwargs)
    elapsed = time.time() - start_time
    self.assertAllGreaterEqual(elapsed, timeout)

  def testNcclHintFallbackToRingReduce(self):
    # [FUNCTION DOC]
    # Skipped when an NCCL all-reduce kernel is registered (GPU builds). Otherwise runs a
    # two-device CPU all-reduce with the NCCL communication hint and expects it to fall back
    # to the default implementation and return the element-wise average.
    """Tests that setting `communication_hint=nccl` works on non-GPU builds."""
    if kernels.get_registered_kernels_for_op('NcclAllReduce'):
      self.skipTest('Run only on non-GPU environments')
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(
          inputs=[[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
                  [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]],
          expected=[0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
          set_graph_key=False,
          communication_hint='nccl')

  def _testWhile(self, num_vars, num_iterations, key_base):
    # [FUNCTION DOC]
    # HELPER: shared driver for the while-loop all-reduce tests.
    # Inputs: num_vars (number of loop variables reduced per device), num_iterations (loop
    # count), key_base (first instance key; later variables use consecutive keys). Enables the
    # scoped-allocator rewrite for the collective reduce op, creates per-device float variables
    # with doubling initial values plus a counter, and runs a while loop whose body all-reduces
    # every variable (after casting to half precision and back) with add and identity.
    # Asserts the final values: each variable has been summed across two devices once per
    # iteration, giving powers of two. Returns nothing.
    group_size = 2
    group_key = 1
    instances = [(key_base + i) for i in range(num_vars)]
    devices = ['CPU:{}'.format(i) for i in range(group_size)]

    config = config_pb2.ConfigProto(device_count={'CPU': group_size})
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')

    with self.session(config=config) as sess:
      loop_vars = []
      for device in devices:
        with ops.device(device):
          loop_vars.append(
              [variable_v1.VariableV1((1 << i) * 1.) for i in range(num_vars)])
      # This variable controls number of iterations.
      loop_vars.append(variable_v1.VariableV1(0.))
      def loop_body(dev0_tensors, dev1_tensors, loop_tensor):
        # [FUNCTION DOC]
        # HELPER: while-loop body for the while tests. Inputs: tensors of device 0, tensors of
        # device 1, and the iteration counter. For every variable on each device casts to half
        # precision, all-reduces (add, identity) and casts back to single precision. Returns the
        # list of the two per-device result lists followed by the counter incremented by one.
        return_ops = []
        for i in range(len(devices)):
          device = devices[i]
          device_tensors = dev0_tensors if i == 0 else dev1_tensors
          with ops.device(device):
            device_collectives = []
            for j in range(num_vars):
              # NOTE(ayushd): we need the `cast` here to ensure that the input
              # to `all_reduce` has an explicit device string.  We don't use
              # `identity` because `cast` is more resilient to getting optimized
              # away by various optimization passes.
              input_tensor = math_ops.cast(device_tensors[j], dtypes.float16)
              collective_op = collective_ops.all_reduce(
                  input_tensor, group_size, group_key, instances[j],
                  'Add', 'Id')
              output_tensor = math_ops.cast(collective_op, dtypes.float32)
              device_collectives.append(output_tensor)
            return_ops.append(device_collectives)
        return_ops.append(math_ops.add(loop_tensor, 1.))
        return return_ops
      # Run until last variable exceeds number of iterations.
      loop_cond = lambda d0, d1, i: math_ops.less(i, num_iterations)
      sess.run(variables.global_variables_initializer())
      results = sess.run(while_loop.while_loop(loop_cond, loop_body, loop_vars))
      self.assertEqual(results[:-1], [
          [((1 << (num_iterations + v)) * 1.) for v in range(num_vars)]
          for _ in range(group_size)])

  def testSimpleWhile(self):
    # [FUNCTION DOC]
    # Runs the while-loop all-reduce helper with one variable and four iterations.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testWhile(num_vars=1, num_iterations=4, key_base=20)

  def testWhileMultipleAllReduce(self):
    # [FUNCTION DOC]
    # Runs the while-loop all-reduce helper with two variables (two reduces per iteration) and
    # four iterations.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testWhile(num_vars=2, num_iterations=4, key_base=20)

  def testWhileWithScopedAllocator(self):
    # [FUNCTION DOC]
    # With the scoped-allocator rewrite enabled for collective reduce, builds on each of two
    # CPUs one input produced by a ten-step while loop and one by an addition, all-reduces both
    # (add, identity) with different instance keys and sums them. Expects both devices to return
    # the value thirty.
    group_size = 2
    group_key = 1
    instance_key0 = 1
    instance_key1 = 2

    config = config_pb2.ConfigProto(device_count={'CPU': group_size})
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')

    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      with self.session(config=config) as sess:
        run_ops = []
        for i in range(group_size):
          with ops.device('CPU:%d' % i):
            constant = constant_op.constant(0.)
            cond = lambda i: math_ops.less(i, 10.)
            body = lambda i: math_ops.add(i, 1.)
            input0 = while_loop.while_loop(cond, body, [constant])
            input1 = math_ops.add(constant, 5)
            colred0 = collective_ops.all_reduce(input0, group_size, group_key,
                                                instance_key0, 'Add', 'Id')
            colred1 = collective_ops.all_reduce(input1, group_size, group_key,
                                                instance_key1, 'Add', 'Id')
            run_ops.append(math_ops.add_n([colred0, colred1]))
        results = sess.run(run_ops)
      self.assertEqual(results, [30., 30.])

  def testCollectiveReduceScalar(self):
    # [FUNCTION DOC]
    # All-reduce of two scalar inputs; expects both devices to return their average.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(inputs=[0.1, 0.3], expected=0.2,
                                 set_graph_key=True)

  def testCollectiveReduceMaximum(self):
    # [FUNCTION DOC]
    # All-reduce with the maximum merge op and identity final op; expects the element-wise
    # maximum of the two inputs on both devices.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(
          inputs=[[1., 20., 3., 40., 5.], [10., 2., 30., 4., 50.]],
          expected=[10., 20., 30., 40., 50.],
          set_graph_key=True,
          instance_key=30,
          merge_op='Max',
          final_op='Id')

  def testCollectiveReduceMinimum(self):
    # [FUNCTION DOC]
    # All-reduce with the minimum merge op and identity final op; expects the element-wise
    # minimum of the two inputs on both devices.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveReduce(
          inputs=[[1., 20., 3., 40., 5.], [10., 2., 30., 4., 50.]],
          expected=[1., 2., 3., 4., 5.],
          set_graph_key=True,
          instance_key=40,
          merge_op='Min',
          final_op='Id')

  def _testCollectiveBroadcast(self, in_val):
    # [FUNCTION DOC]
    # HELPER: shared driver for the broadcast tests. Input: in_val, the value to broadcast
    # (list of numbers or booleans). CPU 0 sends it and CPU 1 receives it with the same shape
    # and type in a two-CPU session with a collective graph key. Asserts that both the sender
    # and receiver outputs are close to the input. Returns nothing.
    group_key = 1
    instance_key = 1
    with self.session(
        config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(in_val)
        out0 = collective_ops.broadcast_send(in0, in0.shape, in0.dtype,
                                             2, group_key, instance_key)
      with ops.device('/CPU:1'):
        c1 = constant_op.constant(in_val)
        out1 = collective_ops.broadcast_recv(c1.shape, c1.dtype,
                                             2, group_key, instance_key)
      run_options = config_pb2.RunOptions()
      run_options.experimental.collective_graph_key = 1
      results = sess.run([out0, out1], options=run_options)
    self.assertAllClose(results[0], in_val, rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], in_val, rtol=1e-5, atol=1e-5)

  def testCollectiveBroadcast(self):
    # [FUNCTION DOC]
    # Broadcasts a float vector between two CPUs via the helper and expects equality.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveBroadcast([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1])

  def testCollectiveBroadcastBool(self):
    # [FUNCTION DOC]
    # Broadcasts a boolean vector between two CPUs via the helper and expects equality.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveBroadcast([True, False])

  def _testCollectiveGather(self, t0, t1, expected, set_graph_key):
    # [FUNCTION DOC]
    # HELPER: shared driver for the all-gather test. Inputs: t0 and t1 (values on CPU 0 and
    # CPU 1), expected (the concatenation both devices should produce along the first axis) and
    # set_graph_key (whether to set a collective graph key). Runs an all-gather on two CPUs and
    # asserts both outputs are close to the expected value. Returns nothing.
    group_key = 1
    instance_key = 1
    with self.session(
        config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(t0)
        c0 = collective_ops.all_gather(in0, 2, group_key, instance_key)
      with ops.device('/CPU:1'):
        in1 = constant_op.constant(t1)
        c1 = collective_ops.all_gather(in1, 2, group_key, instance_key)
      run_options = config_pb2.RunOptions()
      if set_graph_key:
        run_options.experimental.collective_graph_key = 1
      results = sess.run([c0, c1], options=run_options)
    self.assertAllClose(results[0], expected, rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], expected, rtol=1e-5, atol=1e-5)

  def testCollectiveGather(self):
    # [FUNCTION DOC]
    # Runs the all-gather helper on integer inputs of rank one, two and three and expects the
    # first-axis concatenation of the two devices' inputs each time.
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      self._testCollectiveGather([0, 1, 2, 3, 4, 5, 6, 7],
                                 [10, 11, 12, 13, 14, 15, 16, 17],
                                 [0, 1, 2, 3, 4, 5, 6, 7,
                                  10, 11, 12, 13, 14, 15, 16, 17],
                                 True)
      self._testCollectiveGather([[0, 1, 2, 3], [4, 5, 6, 7]],
                                 [[10, 11, 12, 13], [14, 15, 16, 17]],
                                 [[0, 1, 2, 3], [4, 5, 6, 7],
                                  [10, 11, 12, 13], [14, 15, 16, 17]],
                                 True)
      self._testCollectiveGather([[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                                 [[[10, 11], [12, 13]], [[14, 15], [16, 17]]],
                                 [[[0, 1], [2, 3]], [[4, 5], [6, 7]],
                                  [[10, 11], [12, 13]], [[14, 15], [16, 17]]],
                                 True)

  def testCollectiveGatherShapeMismatch(self):
    # [FUNCTION DOC]
    # In one session, first runs an all-gather of equal-shaped inputs successfully, then runs
    # another gather on CPU 1 with a differently shaped input against the same group and
    # instance, and expects an invalid-argument error mentioning a shape mismatch.
    group_key = 1
    instance_key = 1
    t0 = [1, 2, 3, 4]
    t1 = [5, 6, 7, 8]
    t2 = [9, 10]
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      with self.session(
          config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
        with ops.device('/CPU:0'):
          in0 = constant_op.constant(t0)
          c0 = collective_ops.all_gather(in0, 2, group_key, instance_key)
        with ops.device('/CPU:1'):
          in1 = constant_op.constant(t1)
          in2 = constant_op.constant(t2)
          c1 = collective_ops.all_gather(in1, 2, group_key, instance_key)
          c2 = collective_ops.all_gather(in2, 2, group_key, instance_key)
        run_options = config_pb2.RunOptions()
        run_options.experimental.collective_graph_key = 1
        sess.run([c0, c1], options=run_options)
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    'Shape mismatch'):
          sess.run([c0, c2], options=run_options)

  def testCollectiveGatherShapeMismatchAcrossDevices(self):
    # [FUNCTION DOC]
    # All-gather where the two devices supply inputs of different lengths; expects an
    # invalid-argument error mentioning a shape mismatch when the session runs.
    group_key = 1
    instance_key = 1
    t0 = [1, 2, 3, 4]
    t1 = [5, 6]
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      with self.session(
          config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
        with ops.device('/CPU:0'):
          in0 = constant_op.constant(t0)
          c0 = collective_ops.all_gather(in0, 2, group_key, instance_key)
        with ops.device('/CPU:1'):
          in1 = constant_op.constant(t1)
          c1 = collective_ops.all_gather(in1, 2, group_key, instance_key)
        run_options = config_pb2.RunOptions()
        run_options.experimental.collective_graph_key = 1
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    'Shape mismatch'):
          sess.run([c0, c1], options=run_options)

  def testCollectiveGatherPolymorphicShape(self):
    # [FUNCTION DOC]
    # All-gather over placeholders with unknown length, fed twice with different lengths in the
    # same session. Expects each run's output on both devices to be the concatenation of that
    # run's two fed vectors.
    t0 = [0, 1, 2, 3, 4, 5, 6, 7]
    t1 = [10, 11, 12, 13, 14, 15, 16, 17]
    group_size = 2
    group_key = 1
    instance_key = 123
    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      with self.session(
          config=config_pb2.ConfigProto(
              device_count={'CPU': group_size})) as sess:
        with ops.device('/CPU:0'):
          in0 = array_ops.placeholder(dtype=dtypes.int32, shape=[None])
          c0 = collective_ops.all_gather(in0, group_size, group_key,
                                         instance_key)
        with ops.device('/CPU:1'):
          in1 = array_ops.placeholder(dtype=dtypes.int32, shape=[None])
          c1 = collective_ops.all_gather(in1, group_size, group_key,
                                         instance_key)

        results = sess.run([c0, c1], feed_dict={in0: t0, in1: t1})
        results_ = sess.run([c0, c1], feed_dict={in0: t0[1:], in1: t1[1:]})

    expected_output = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    self.assertAllClose(results[0], expected_output, rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], expected_output, rtol=1e-5, atol=1e-5)

    expected_output_ = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17]
    self.assertAllClose(results_[0], expected_output_, rtol=1e-5, atol=1e-5)
    self.assertAllClose(results_[1], expected_output_, rtol=1e-5, atol=1e-5)

  @test_util.run_v2_only
  @test_util.disable_tfrt(
      'b/177270918: TFRT has dead lock when executing collective ops.')
  def testCollectiveGroupSizeMismatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, this test never runs a collective or asserts a group size
    # mismatch. It only checks that one physical CPU exists, splits it into two logical CPUs
    # and initializes the context. The mismatch scenario (group sizes two and three) lives in a
    # nested function inside the shape-check-failure test below.
    # Runs only in v2 mode and with the TFRT runtime disabled.
    cpus = config.list_physical_devices('CPU')
    self.assertEqual(len(cpus), 1)
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])
    context.ensure_initialized()

  @test_util.run_v2_only
  def testCollectiveGatherShapeCheckFailure(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test mixes two scenarios. Besides the gather input-rank check named
    # here, it also defines and runs the group-size mismatch check that its name suggests
    # belongs to the previous test; that part needs two logical CPUs, which are only configured
    # in that previous test, so it relies on test ordering or context state.
    # Runs only in v2 mode. First calls the raw gather op with a scalar input and expects an
    # invalid-argument error about rank. Then runs a compiled function that all-reduces with
    # claimed group sizes two and three under the same keys and expects an internal error that
    # the group has a different size.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'input should have rank > 0'):
      collective_ops.gen_collective_ops.CollectiveGather(
          input=1,
          group_size=1,
          group_key=1,
          instance_key=1,
          shape=(3, 3, 3),
          communication_hint='auto',
          timeout_seconds=0,
          name='')

    @def_function.function
    def run_all_reduce():
      # [FUNCTION DOC]
      # HELPER: compiled function used to provoke a group-size mismatch. Takes no arguments;
      # all-reduces a vector on CPU 0 declaring group size two and another on CPU 1 declaring
      # group size three, with the same group and instance keys. Returns both results.
      group_key = 10
      instance_key = 20
      t0 = [1, 2, 3, 4]
      t1 = [5, 6, 7, 8]
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(t0)
        c0 = collective_ops.all_reduce(
            in0, group_size=2, group_key=group_key, instance_key=instance_key,
            merge_op='Add', final_op='Id')
      with ops.device('/CPU:1'):
        in1 = constant_op.constant(t1)
        c1 = collective_ops.all_reduce(
            in1, group_size=3, group_key=group_key, instance_key=instance_key,
            merge_op='Add', final_op='Id')
      return c0, c1

    with self.assertRaisesRegex(errors.InternalError,
                                'but that group has size'):
      run_all_reduce()

  @test_util.run_v2_only
  def testCollectiveTensorsHaveNoDeviceSpecified(self):
    # [FUNCTION DOC]
    # Runs only in v2 mode. Splits the single physical CPU into two logical CPUs, then calls a
    # compiled function with one input placed on each. Inside the function (at trace time) it
    # asserts the inputs have no device string, and all-reduces them under per-device scopes.
    # Expects both outputs to equal the average of one and three, i.e. two.
    cpus = config.list_physical_devices('CPU')
    self.assertEqual(len(cpus), 1)
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])
    context.ensure_initialized()

    group_size = 2
    group_key = 1
    instance_key = 1

    @def_function.function
    def fn(all_args):
      # [FUNCTION DOC]
      # HELPER: compiled function for the no-device test. Input: a list of two tensors with no
      # device assigned. Asserts at trace time that neither has a device, then all-reduces the
      # first on CPU 0 and the second on CPU 1 (add, divide). Returns the list of two results.
      results = []
      # The inputs have no devices set. This is expected to be a trace-time
      # check only.
      self.assertEqual(all_args[0].device, '')
      self.assertEqual(all_args[1].device, '')

      with ops.device('/CPU:0'):
        results.append(
            collective_ops.all_reduce(all_args[0], group_size, group_key,
                                      instance_key, 'Add', 'Div'))
      with ops.device('/CPU:1'):
        results.append(
            collective_ops.all_reduce(all_args[1], group_size, group_key,
                                      instance_key, 'Add', 'Div'))

      return results

    with ops.device('/CPU:0'):
      in0 = constant_op.constant(1)
    with ops.device('/CPU:1'):
      in1 = constant_op.constant(3)
    result = fn([in0, in1])
    self.assertAllClose(result, [2, 2])

  def testConstantWithScopedAllocator(self):
    # [FUNCTION DOC]
    # With constant folding and the scoped-allocator rewrite for collective reduce enabled,
    # derives two identity tensors from one constant per CPU and all-reduces both (add, identity)
    # with different instance keys. Expects all four results to be three (one plus two).
    group_size = 2
    group_key = 1
    instance_key1 = 1
    instance_key2 = 2

    graph_options = config_pb2.GraphOptions(
        optimizer_options=config_pb2.OptimizerOptions(do_constant_folding=True))
    cfg = config_pb2.ConfigProto(device_count={'CPU': group_size},
                                 graph_options=graph_options)
    rewrite_options = cfg.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')

    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default():
      with self.session(config=cfg) as sess:
        run_ops = []
        for i in range(group_size):
          with ops.device('CPU:%d' % i):
            constant = constant_op.constant(i + 1.)
            input_tensor1 = array_ops.identity(constant)
            input_tensor2 = array_ops.identity(constant)
            reduced_tensor1 = collective_ops.all_reduce(
                input_tensor1, group_size, group_key, instance_key1, 'Add',
                'Id')
            reduced_tensor2 = collective_ops.all_reduce(
                input_tensor2, group_size, group_key, instance_key2, 'Add',
                'Id')
            run_ops.append(array_ops.identity(reduced_tensor1))
            run_ops.append(array_ops.identity(reduced_tensor2))
        results = sess.run(run_ops)
    self.assertEqual(results, [3., 3., 3., 3.])


if __name__ == '__main__':
  test.main()
