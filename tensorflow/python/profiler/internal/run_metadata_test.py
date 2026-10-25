# [MODULE DOC]
# Tests that the RunMetadata proto collected during a traced session run is consistent with the
# profiler's model analysis output. Contains module-level helpers that extract per-device node
# statistics and that run a matmul model or a looping RNN model with software tracing, plus a
# test class covering CPU, GPU, loop and gradient-graph scenarios. Uses the shared profiler
# test library for model construction. Run through the standard TensorFlow test main.
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
"""test the RunMetadata proto."""

from collections import defaultdict

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.profiler import option_builder

# pylint: disable=g-bad-import-order
# XXX: this depends on pywrap_tensorflow and must come later
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib

SIZE = 1300
builder = option_builder.ProfileOptionBuilder


def _extract_node(run_meta, node_name):
  # [FUNCTION DOC]
  # HELPER: collects execution statistics for one node from run metadata in tests.
  # Walks all device statistics in the step stats of the given run metadata, normalizes each
  # device name to a short form starting at the cpu or gpu portion (host cpu devices are kept
  # with their full lowercase name), and fails an assertion for unrecognized device names.
  # Node names are truncated at the first colon. Returns a mapping from normalized device name
  # to the list of node statistics whose name equals node_name (default-empty lists).
  ret = defaultdict(list)
  for dev_stat in run_meta.step_stats.dev_stats:
    dev = dev_stat.device.lower()
    if dev.find('cpu:') > 0:
      dev = dev[dev.find('cpu:'):]
    elif dev.find('gpu:') > 0:
      dev = dev[dev.find('gpu:'):]
    elif '/host:cpu' not in dev:
      assert False, 'Unrecognized device name: %s' % dev

    for node_stat in dev_stat.node_stats:
      nname = node_stat.node_name
      if nname.find(':') > 0:
        nname = nname[:nname.find(':')]
      if nname == node_name:
        ret[dev].append(node_stat)
  return ret


def _run_model():
  # [FUNCTION DOC]
  # HELPER: runs a simple model under tracing for the tests.
  # Builds a random 1xSIZE by SIZEx2*SIZE matrix multiplication in the default graph, runs it in
  # a new session with the arithmetic optimizer turned off and software tracing, then profiles
  # the graph with the run metadata using time-and-memory options sorted by name with no
  # output printed. Returns a pair of the profile root node and the collected run metadata.
  x = random_ops.random_normal(shape=[1, SIZE])
  w = random_ops.random_normal(shape=[SIZE, 2 * SIZE])
  y = math_ops.matmul(x, w)

  config = config_pb2.ConfigProto()
  config.graph_options.rewrite_options.arithmetic_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)
  with session.Session(config=config) as sess:
    run_metadata = config_pb2.RunMetadata()
    opts = builder.time_and_memory()
    opts['min_micros'] = 0
    opts['min_bytes'] = 0
    opts['order_by'] = 'name'
    opts['output'] = 'none'
    _ = sess.run(
        y,
        options=config_pb2.RunOptions(
            trace_level=config_pb2.RunOptions.SOFTWARE_TRACE),
        run_metadata=run_metadata)
    tfprof_node = model_analyzer.profile(
        sess.graph, run_meta=run_metadata, options=opts)

    return tfprof_node, run_metadata


def _run_loop_model():
  # [FUNCTION DOC]
  # HELPER: runs the full RNN training model under tracing for the tests.
  # Opens a session with the remapper optimization disabled (so matmul is not fused with bias
  # add), builds the full model from the test library, initializes global variables, runs the
  # train op once with software tracing, then profiles the graph with time-and-memory options
  # sorted by name and no output. Returns a pair of the profile root node and the run metadata.
  config = config_pb2.ConfigProto()
  # Grappler might fuse MatMul with BiasAdd in remapper optimizer.
  config.graph_options.rewrite_options.remapping = (
      rewriter_config_pb2.RewriterConfig.OFF)
  with session.Session(config=config) as sess:
    x = lib.BuildFullModel()

    sess.run(variables.global_variables_initializer())
    run_meta = config_pb2.RunMetadata()
    _ = sess.run(
        x,
        options=config_pb2.RunOptions(
            trace_level=config_pb2.RunOptions.SOFTWARE_TRACE),
        run_metadata=run_meta)

    opts = builder.time_and_memory()
    opts['order_by'] = 'name'
    opts['output'] = 'none'

    tfprof_node = model_analyzer.profile(sess.graph, run_meta, options=opts)
    return tfprof_node, run_meta


class RunMetadataTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that run metadata from traced runs, and the profile built from it,
  # contain the expected nodes, devices, allocation records and run counts on CPU and GPU, plus
  # a structural check of gradient op names. GPU tests return early when no GPU is available.
  # Fixtures are the module-level model runners; graphs are reset at the start of each test.

  # This test requires HARDWARE_TRACE or FULL_TRACE to be specified to
  # work as expected. Since we now run this test with SOFTWARE_TRACE
  # (see _run_model routine above), this test will / should fail since
  # GPU device tracers are not enabled
  @test_util.run_deprecated_v1
  def testGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment above says this test should fail with software tracing because GPU
    # tracers are disabled, yet it asserts success; and it silently passes without a GPU.
    # Scenario: run the matmul model on the GPU device (returns immediately if no CUDA GPU).
    # Expected: first child of the profile is named MatMul with execution time above 10
    # microseconds, and exactly one MatMul statistic exists for device gpu:0.
    if not test.is_gpu_available(cuda_only=True):
      return

    gpu_dev = test.gpu_device_name()
    ops.reset_default_graph()
    with ops.device(gpu_dev):
      tfprof_node, run_meta = _run_model()
      self.assertEqual(tfprof_node.children[0].name, 'MatMul')
      self.assertGreater(tfprof_node.children[0].exec_micros, 10)

    ret = _extract_node(run_meta, 'MatMul')
    self.assertEqual(len(ret['gpu:0']), 1)

  @test_util.run_deprecated_v1
  def testAllocationHistory(self):
    # [FUNCTION DOC]
    # Scenario: run the matmul model on the GPU (returns immediately without a CUDA GPU).
    # Expected for the MatMul node: exactly two allocation records, an allocation followed by a
    # deallocation with later timestamp, positive then negative bytes summing to zero. For the
    # random normal node: its allocation happens before the matmul starts and its deallocation
    # after the matmul started.
    if not test.is_gpu_available(cuda_only=True):
      return

    gpu_dev = test.gpu_device_name()
    ops.reset_default_graph()
    with ops.device(gpu_dev):
      _, run_meta = _run_model()

    mm = _extract_node(run_meta, 'MatMul')['gpu:0'][0]
    mm_allocs = mm.memory[0].allocation_records
    # has allocation and deallocation.
    self.assertEqual(len(mm_allocs), 2)
    # first allocated.
    self.assertGreater(mm_allocs[1].alloc_micros, mm_allocs[0].alloc_micros)
    self.assertGreater(mm_allocs[0].alloc_bytes, 0)
    # Then deallocated.
    self.assertLess(mm_allocs[1].alloc_bytes, 0)
    # All memory deallocated.
    self.assertEqual(mm_allocs[0].alloc_bytes + mm_allocs[1].alloc_bytes, 0)

    rand = _extract_node(run_meta,
                         'random_normal/RandomStandardNormal')['gpu:0'][0]
    random_allocs = rand.memory[0].allocation_records
    # random normal must allocated first since matmul depends on it.
    self.assertLess(random_allocs[0].alloc_micros, mm.all_start_micros)
    # deallocates the memory after matmul started.
    self.assertGreater(random_allocs[1].alloc_micros, mm.all_start_micros)

  @test_util.run_deprecated_v1
  def testCPU(self):
    # [FUNCTION DOC]
    # Scenario: run the matmul model on cpu:0. Expected: the first child of the profile is
    # MatMul with positive execution time, exactly one MatMul statistic on device cpu:0, and no
    # statistics found for the name with a colon-repeated suffix (MatMul:MatMul).
    ops.reset_default_graph()
    with ops.device('/cpu:0'):
      tfprof_node, run_meta = _run_model()
      self.assertEqual(tfprof_node.children[0].name, 'MatMul')
      self.assertGreater(tfprof_node.children[0].exec_micros, 0)

    ret = _extract_node(run_meta, 'MatMul')
    self.assertEqual(len(ret['cpu:0']), 1)

    ret = _extract_node(run_meta, 'MatMul:MatMul')
    self.assertEqual(len(ret), 0)

  @test_util.run_v1_only('b/120545219')
  def testLoopCPU(self):
    # [FUNCTION DOC]
    # Scenario: run the RNN model on cpu:0 (graph mode only). The while loop makes the RNN cell
    # matmul execute four times. Expected: four node statistics on cpu:0; the profile node found
    # by name has run count four and both its cpu and total execution times equal the sum of the
    # relative op end times from the run metadata.
    ops.reset_default_graph()
    with ops.device('/cpu:0'):
      tfprof_node, run_meta = _run_loop_model()
      # The while-loop caused a node to appear 4 times in scheduling.
      ret = _extract_node(run_meta, 'rnn/while/basic_rnn_cell/MatMul')
      self.assertEqual(len(ret['cpu:0']), 4)

      total_cpu_execs = 0
      for node in ret['cpu:0']:
        total_cpu_execs += node.op_end_rel_micros

      mm_node = lib.SearchTFProfNode(tfprof_node,
                                     'rnn/while/basic_rnn_cell/MatMul')

      self.assertEqual(mm_node.run_count, 4)
      self.assertEqual(mm_node.cpu_exec_micros, total_cpu_execs)
      self.assertEqual(mm_node.exec_micros, total_cpu_execs)

  def testGradientGraph(self):
    # [FUNCTION DOC]
    # Scenario: build the RNN model graph (running it once) and classify operations as backward
    # if their name contains a gradients scope followed by a grad scope, otherwise forward.
    # Expected: the forward-op name embedded between those markers in every backward op name
    # exists among the forward ops. The profiler's code view depends on this naming convention.
    # Note: Please don't just adjust the test to make it pass.
    # The code view logic depends on it.
    ops.reset_default_graph()
    _, _ = _run_loop_model()
    graph = ops.get_default_graph()
    forward_op = set()
    backward_op = set()
    back_to_forward = {}
    for op in graph.get_operations():
      if op.name.find('gradients/') > 0 and op.name.find('_grad/') > 0:
        backward_op.add(op.name)
        idx1 = op.name.find('gradients/') + 10
        idx2 = op.name.find('_grad/')
        back_to_forward[op.name] = op.name[idx1:idx2]
      else:
        forward_op.add(op.name)

    for _, f in back_to_forward.items():
      self.assertTrue(f in forward_op)

  # This test requires HARDWARE_TRACE or FULL_TRACE to be specified to
  # work as expected. Since we now run this test with SOFTWARE_TRACE
  # (see _run_model routine above), this test will / should fail since
  # GPU device tracers are not enabled
  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Test fails on ROCm when run without FULL_TRACE')
  def testLoopGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the stream assertion is indented inside the loop over nodes, so it runs once
    # per statistic rather than once, and the comment above says the test should fail with
    # software tracing while it asserts success. Returns silently if no GPU exists.
    # Scenario: run the RNN model on GPU 0. Expected: four MatMul statistics on gpu:0 and at
    # least four entries for the all-streams gpu stream; skipped on ROCm builds.
    if not test.is_gpu_available():
      return

    ops.reset_default_graph()
    with ops.device('/device:GPU:0'):
      _, run_meta = _run_loop_model()
      # The while-loop caused a node to appear 4 times in scheduling.
      ret = _extract_node(run_meta, 'rnn/while/basic_rnn_cell/MatMul')
      self.assertEqual(len(ret['gpu:0']), 4, '%s' % run_meta)

      total_cpu_execs = 0
      for node in ret['gpu:0']:
        total_cpu_execs += node.op_end_rel_micros

        self.assertGreaterEqual(
            len(ret['gpu:0/stream:all']), 4, '%s' % run_meta)


if __name__ == '__main__':
  test.main()
