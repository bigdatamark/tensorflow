# [MODULE DOC]
# Unit tests for the legacy profiler model analyzer API (profile, advise and the Profiler class)
# together with the option builder and profile context. The tests build small and full models
# using a shared test library, run them under a session with full tracing, and check the text
# reports (written to files), returned profile protos, timeline and pprof outputs, minimum
# thresholds, attribute selection, automatic periodic profiling and out-of-memory reports.
# Several tests only run on GPU machines or only in graph mode. Contains a single test class
# and a main entry that runs the test runner.
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

import gzip
import io
import os
import random
import re

import numpy as np

from tensorflow.core.profiler import profile_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import profile_context
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib
from tensorflow.python.util import compat

builder = option_builder.ProfileOptionBuilder


class PrintModelAnalysisTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the model analyzer profiling functions and the profiler option builder
  # end to end. Shared fixture: a session configuration that disables host-pinning graph
  # rewrites so that node placement and names are predictable. Tests write reports to
  # temporary files and inspect their contents or the returned profile protos.

  def _no_rewrite_session_config(self):
    # [FUNCTION DOC]
    # HELPER: Shared fixture factory for the session configurations used by the tests.
    # Builds and returns a session configuration whose graph rewriter has the pin-to-host
    # optimization turned off. No inputs, no side effects.
    rewriter_config = rewriter_config_pb2.RewriterConfig(
        pin_to_host_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    return config_pb2.ConfigProto(graph_options=graph_options)

  def testDumpToFile(self):
    # [FUNCTION DOC]
    # Scenario: profile a small model's trainable variable parameters with file output.
    # Resets the default graph, builds options writing to a temporary dump file, builds the
    # small model and runs the profile. Asserts the file content (after stripping the
    # documentation header) lists the root node with 451 parameters and the three variables
    # DW, DW2 and ScalarW with their shapes and parameter counts.
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = builder(builder.trainable_variables_parameter()).with_file_output(
        outfile).build()

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      _ = lib.BuildSmallModel()
      model_analyzer.profile(sess.graph, options=opts)

      with gfile.Open(outfile, 'r') as f:
        self.assertEqual(
            u'node name | # parameters\n'
            '_TFProfRoot (--/451 params)\n'
            '  DW (3x3x3x6, 162/162 params)\n'
            '  DW2 (2x2x6x12, 288/288 params)\n'
            '  ScalarW (1, 1/1 params)\n', lib.CheckAndRemoveDoc(f.read()))

  @test_util.run_v1_only('b/120545219')
  def testSelectEverythingDetail(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): select every available metric using a profile context that
    # traces and dumps the next step, on GPU if available otherwise CPU. Profiles by name scope
    # into a file and asserts the header line lists all selected columns. For convolution rows
    # it asserts that time is positive, the device is the expected kind, float operations are
    # positive, the op count is one run over one defined and the input shapes are as expected;
    # for the DW variable row it asserts positive parameter counts. Finally removes the file,
    # reloads the saved profile with a profiler restored from file, re-runs the query and asserts
    # the new output equals the earlier one.
    ops.reset_default_graph()
    dev = '/device:GPU:0' if test.is_gpu_available() else '/device:CPU:0'
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (
        builder(builder.trainable_variables_parameter()).with_file_output(
            outfile).with_accounted_types(['.*']).select([
                'micros', 'bytes', 'params', 'float_ops', 'occurrence',
                'device', 'op_types', 'input_shapes'
            ]).build())

    with profile_context.ProfileContext(
        test.get_temp_dir(), trace_steps=[], dump_steps=[]) as pctx:
      with session.Session(
          config=self._no_rewrite_session_config()) as sess, ops.device(dev):
        x = lib.BuildSmallModel()

        self.evaluate(variables.global_variables_initializer())
        pctx.trace_next_step()
        pctx.dump_next_step()
        _ = self.evaluate(x)

        pctx.profiler.profile_name_scope(options=opts)

        with gfile.Open(outfile, 'r') as f:
          # pylint: disable=line-too-long
          dump_str = lib.CheckAndRemoveDoc(f.read())
          outputs = dump_str.split('\n')

          self.assertEqual(
              outputs[0],
              'node name | # parameters | # float_ops | requested bytes | total execution time | accelerator execution time | cpu execution time | assigned devices | op types | op count (run|defined) | input shapes'
          )
          for o in outputs[1:]:
            if o.find('Conv2D ') > 0:
              metrics = o[o.find('(') + 1:o.find(')')].split(',')
              # Make sure time is profiled.
              gap = 1 if test.is_gpu_available() else 2
              for i in range(3, 6, gap):
                mat = re.search('(.*)(?:us|ms|sec)/(.*)(?:us|ms|sec)',
                                metrics[i])
                self.assertGreater(float(mat.group(1)), 0.0)
                self.assertGreater(float(mat.group(2)), 0.0)
              # Make sure device is profiled.
              if test.is_gpu_available():
                self.assertTrue(metrics[6].find('gpu') > 0)
                self.assertFalse(metrics[6].find('cpu') > 0)
              else:
                self.assertFalse(metrics[6].find('gpu') > 0)
                self.assertTrue(metrics[6].find('cpu') > 0)
              # Make sure float_ops is profiled.
              mat = re.search('(.*)k/(.*)k flops', metrics[1].strip())
              self.assertGreater(float(mat.group(1)), 0.0)
              self.assertGreater(float(mat.group(2)), 0.0)
              # Make sure op_count is profiled.
              self.assertEqual(metrics[8].strip(), '1/1|1/1')
              # Make sure input_shapes is profiled.
              self.assertEqual(metrics[9].strip(), '0:2x6x6x3|1:3x3x3x6')

            if o.find('DW (3x3x3x6') > 0:
              metrics = o[o.find('(') + 1:o.find(')')].split(',')
              mat = re.search('(.*)/(.*) params', metrics[1].strip())
              self.assertGreater(float(mat.group(1)), 0.0)
              self.assertGreater(float(mat.group(2)), 0.0)
          # pylint: enable=line-too-long

    # Test that profiler restored from profile file gives the same result.
    gfile.Remove(outfile)
    profile_file = os.path.join(test.get_temp_dir(), 'profile_1')
    with lib.ProfilerFromFile(profile_file) as profiler:
      profiler.profile_name_scope(options=opts)
      with gfile.Open(outfile, 'r') as f:
        self.assertEqual(dump_str, lib.CheckAndRemoveDoc(f.read()))

  def testSelectEverything(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test makes no assertions; it only checks that profiling with many selected
    # attributes on CPU with full tracing does not raise.
    # Scenario: run the small model with full tracing under a CPU device, then call profile with
    # options selecting parameters, float ops, occurrence, device, op types and input shapes.
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (
        builder(builder.trainable_variables_parameter()).with_file_output(
            outfile).with_accounted_types(['.*']).select([
                'params', 'float_ops', 'occurrence', 'device', 'op_types',
                'input_shapes'
            ]).build())

    with session.Session(config=self._no_rewrite_session_config()
                        ) as sess, ops.device('/device:CPU:0'):
      x = lib.BuildSmallModel()

      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      model_analyzer.profile(sess.graph, run_meta, options=opts)

  def testSimpleCodeView(self):
    # [FUNCTION DOC]
    # Scenario: profile the small model with the code view, showing only nodes from the test
    # library file and selecting bytes, parameters, float ops, hidden ops, device and input
    # shapes, writing to a file. Asserts that the first 80 characters of the report (after the
    # documentation header is removed) equal the expected column header text.
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    # TODO(xpan): Test 'micros'. Since the execution time changes each run,
    # it's a bit difficult to test it now.
    opts = (
        builder(builder.trainable_variables_parameter()).with_file_output(
            outfile).with_accounted_types(['.*']).with_node_names(
                show_name_regexes=['.*model_analyzer_testlib.*'
                                  ]).account_displayed_op_only(False).select([
                                      'bytes', 'params', 'float_ops',
                                      'num_hidden_ops', 'device', 'input_shapes'
                                  ]).build())

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildSmallModel()

      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      model_analyzer.profile(sess.graph, run_meta, cmd='code', options=opts)

      with gfile.Open(outfile, 'r') as f:
        # pylint: disable=line-too-long
        self.assertEqual(
            'node name | requested bytes | # parameters | # float_ops | assigned devices | in',
            lib.CheckAndRemoveDoc(f.read())[0:80])
        # pylint: enable=line-too-long

  @test_util.run_v1_only('b/120545219')
  def testComplexCodeView(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected child names embed hard-coded line numbers of another file, so any
    # edit of that library file breaks this test.
    # Scenario (graph mode only): profile the full model in Python code view through a profile
    # context tracing the next step, selecting parameters and float ops. Asserts the report file
    # has more than five lines and starts with the expected header, and the returned proto has
    # positive execution time, 2844 total parameters, float operations above a threshold that
    # depends on whether the MKL build is enabled, ten children, root name, and the exact child
    # names giving source line numbers of the test library file (including gradient entries).
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (
        builder(builder.trainable_variables_parameter()).with_file_output(
            outfile).with_accounted_types(['.*']).with_node_names(
                show_name_regexes=['.*model_analyzer_testlib.py.*'
                                  ]).account_displayed_op_only(False).select(
                                      ['params', 'float_ops']).build())

    with profile_context.ProfileContext(
        test.get_temp_dir(), trace_steps=[], dump_steps=[]) as pctx:
      with session.Session(config=self._no_rewrite_session_config()) as sess:
        x = lib.BuildFullModel()

        self.evaluate(variables.global_variables_initializer())
        pctx.trace_next_step()
        _ = self.evaluate(x)
        tfprof_node = pctx.profiler.profile_python(options=opts)

        # pylint: disable=line-too-long
        with gfile.Open(outfile, 'r') as f:
          lines = f.read().split('\n')
          self.assertGreater(len(lines), 5)
          result = '\n'.join(l[:min(len(l), 80)] for l in lines)
          self.assertTrue(
              compat.as_text(lib.CheckAndRemoveDoc(result)).startswith(
                  'node name | # parameters | # float_ops'))

        self.assertLess(0, tfprof_node.total_exec_micros)
        self.assertEqual(2844, tfprof_node.total_parameters)
        #The graph is modified when MKL is enabled,total_float_ops will
        #be different
        if test_util.IsMklEnabled():
          self.assertLess(101600, tfprof_node.total_float_ops)
        else:
          self.assertLess(145660, tfprof_node.total_float_ops)
        self.assertEqual(10, len(tfprof_node.children))
        self.assertEqual('_TFProfRoot', tfprof_node.name)
        self.assertEqual('model_analyzer_testlib.py:59:BuildFullModel',
                         tfprof_node.children[0].name)
        self.assertEqual(
            'model_analyzer_testlib.py:59:BuildFullModel (gradient)',
            tfprof_node.children[1].name)
        self.assertEqual('model_analyzer_testlib.py:62:BuildFullModel',
                         tfprof_node.children[2].name)
        self.assertEqual(
            'model_analyzer_testlib.py:62:BuildFullModel (gradient)',
            tfprof_node.children[3].name)
        self.assertEqual('model_analyzer_testlib.py:63:BuildFullModel',
                         tfprof_node.children[4].name)
        self.assertEqual('model_analyzer_testlib.py:63:BuildFullModel (gradient)',
                         tfprof_node.children[5].name)
        self.assertEqual(
            'model_analyzer_testlib.py:65:BuildFullModel',
            tfprof_node.children[6].name)
        self.assertEqual('model_analyzer_testlib.py:66:BuildFullModel',
                         tfprof_node.children[7].name)
        # pylint: enable=line-too-long

  def testCodeViewLeafGraphNode(self):
    # [FUNCTION DOC]
    # Scenario: profile the small model with the code view and empty output. Walks down the first
    # child chain of the returned tree, asserting that inner nodes hold no graph nodes while the
    # leaf holds exactly one graph node.
    ops.reset_default_graph()
    opts = (
        builder(builder.trainable_variables_parameter()).with_empty_output()
        .with_accounted_types(['.*']).account_displayed_op_only(False).select(
            ['bytes', 'params', 'float_ops', 'device']).build())

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildSmallModel()

      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta, cmd='code', options=opts)

      leaf = tfprof_node
      while leaf.children:
        self.assertEqual(0, len(leaf.graph_nodes))
        leaf = leaf.children[0]
      self.assertEqual(1, len(leaf.graph_nodes))

  def testTimeline(self):
    # [FUNCTION DOC]
    # Scenario: profile the full model in graph view with timeline output for step zero and a
    # very large depth. Asserts that a timeline file with the step suffix is created and has
    # content (more than 1000 characters on non-Windows systems, at least 2 on Windows).
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'timeline')
    opts = (
        builder(builder.trainable_variables_parameter()).with_max_depth(100000)
        .with_step(0).with_timeline_output(outfile).with_accounted_types(
            ['.*']).build())

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildFullModel()

      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      _ = model_analyzer.profile(
          sess.graph, run_meta, cmd='graph', options=opts)

      with gfile.Open(outfile + '_0', 'r') as f:
        # Test that a json file is created.
        # TODO(xpan): tfprof Timeline isn't quite correct on Windows.
        # Investigate why.
        if os.name != 'nt':
          self.assertLess(1000, len(f.read()))
        else:
          self.assertLess(1, len(f.read()))

  def testOpView(self):
    # [FUNCTION DOC]
    # Scenario: profile the full model with the operation-type view, minimum occurrence 10,
    # ordered by occurrence, selecting many metrics, written to a file. Asserts the whitespace
    # stripped header text matches the expected columns, then walks the single-child chain of the
    # returned tree asserting execution time accounting is consistent between parent and child,
    # occurrence counts are non-increasing, and some input shapes were recorded.
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')

    opts = (
        builder(builder.trainable_variables_parameter()).with_file_output(
            outfile).with_accounted_types(
                ['.*']).with_min_occurrence(10).order_by('occurrence').select([
                    'params', 'micros', 'bytes', 'peak_bytes', 'residual_bytes',
                    'output_bytes', 'occurrence', 'input_shapes'
                ]).build())

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildFullModel()

      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta, cmd='op', options=opts)

      with gfile.Open(outfile, 'r') as f:
        # pylint: disable=line-too-long
        self.assertEqual(
            'nodename|requestedbytes|peakbytes|residualbytes|outputbytes|totalexecutiontime|acceleratorexecutiontime|cpuexecutiontime|#parameters|opoccurrence(run|defined)|inputshapes',
            lib.CheckAndRemoveDoc(f.read()).replace('\t',
                                                    '').replace(' ', '')[0:170])
        # pylint: enable=line-too-long

      total_children = 0
      last_occurrence = 1e32
      input_shapes = 0
      last_total_micros = tfprof_node.total_exec_micros
      last_micros = tfprof_node.exec_micros
      while tfprof_node.children:
        for gnode in tfprof_node.graph_nodes:
          input_shapes += len(gnode.input_shapes)
        self.assertEqual(len(tfprof_node.children), 1)
        tfprof_node = tfprof_node.children[0]

        self.assertEqual(last_total_micros,
                         tfprof_node.total_exec_micros + last_micros)
        last_total_micros = tfprof_node.total_exec_micros
        last_micros = tfprof_node.exec_micros

        total_children += 1
        self.assertLessEqual(len(tfprof_node.graph_nodes), last_occurrence)
        last_occurrence = len(tfprof_node.graph_nodes)

      self.assertGreater(input_shapes, 0)

  def testAdvisor(self):
    # [FUNCTION DOC]
    # Scenario: run the full model with full tracing and call advise. Asserts the returned
    # advice contains the accelerator utilization, expensive operation and operation checkers;
    # the accelerator checker has reports only when a GPU is available, and the expensive
    # operation checker always has at least one report.
    ops.reset_default_graph()

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildFullModel()

      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      advice_pb = model_analyzer.advise(sess.graph, run_meta)
      self.assertTrue('AcceleratorUtilizationChecker' in advice_pb.checkers)
      self.assertTrue('ExpensiveOperationChecker' in advice_pb.checkers)
      self.assertTrue('OperationChecker' in advice_pb.checkers)

      checker = advice_pb.checkers['AcceleratorUtilizationChecker']
      if test.is_gpu_available():
        self.assertGreater(len(checker.reports), 0)
      else:
        self.assertEqual(len(checker.reports), 0)
      checker = advice_pb.checkers['ExpensiveOperationChecker']
      self.assertGreater(len(checker.reports), 0)

  def pprof_test_helper(self, attribute, should_fail=False):
    # [FUNCTION DOC]
    # HELPER: Shared body for the pprof output test.
    # Inputs: attribute, the profile attribute to select (string); should_fail, boolean, true
    # when the attribute is unsupported for pprof output (default false). Profiles the full
    # model in code view with pprof output to a temporary gzip file, trimming nodes from the ops
    # module. If failure is expected asserts the output file was not created and stops.
    # Otherwise decompresses and parses the profile and asserts it has more than ten samples,
    # locations and functions, more than thirty strings, includes an rnn string and no string
    # starting with the trimmed module name.
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), attribute + '_pprof.pb.gz')
    opts = (
        builder(builder.time_and_memory()).select([
            attribute
        ]).with_max_depth(100000).with_node_names(
            trim_name_regexes=['ops.py.*']).with_pprof_output(outfile).build())

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildFullModel()

      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      _ = model_analyzer.profile(sess.graph, run_meta, cmd='code', options=opts)

      if should_fail:
        self.assertFalse(gfile.Exists(outfile))
        return

      profile_pb = profile_pb2.Profile()
      with gfile.Open(outfile, 'rb') as f:
        with gzip.GzipFile(fileobj=io.BytesIO(f.read())) as gzipf:
          profile_pb.ParseFromString(gzipf.read())

      self.assertGreater(len(profile_pb.sample), 10)
      self.assertGreater(len(profile_pb.location), 10)
      self.assertGreater(len(profile_pb.function), 10)
      self.assertGreater(len(profile_pb.string_table), 30)

      has_rnn = False
      for s in profile_pb.string_table:
        if s.find('rnn') > 0:
          has_rnn = True
        self.assertFalse(s.startswith('ops.py'))
      self.assertTrue(has_rnn)

  def testPprof(self):
    # [FUNCTION DOC]
    # Scenario: run the pprof helper for each supported attribute (micros, bytes, accelerator
    # micros, cpu micros, params, float ops) expecting a valid pprof file, and for the
    # unsupported attributes (op types, device, input shapes) expecting no output file.
    for attr in [
        'micros', 'bytes', 'accelerator_micros', 'cpu_micros', 'params',
        'float_ops'
    ]:
      self.pprof_test_helper(attr)
    for attr in ['op_types', 'device', 'input_shapes']:
      self.pprof_test_helper(attr, True)

  def testMinOption(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses an unseeded random threshold, so runs are not reproducible.
    # Scenario: for each minimum threshold option (execution time, accelerator time, cpu time,
    # requested bytes, peak bytes, residual bytes, output bytes) set to the same random value
    # between 0 and 10000, profile the small model and assert that every returned node,
    # recursively, has the corresponding metric at least the threshold.
    ops.reset_default_graph()

    def check_min(nodes, mm=0, mam=0, mcm=0, mb=0, mpb=0, mrb=0, mob=0):
      # [FUNCTION DOC]
      # HELPER: Recursive assertion helper for the minimum option test.
      # Inputs: nodes, list of profile nodes; mm, mam, mcm, mb, mpb, mrb, mob are the minimum
      # execution micros, accelerator micros, cpu micros, requested bytes, peak bytes, residual
      # bytes and output bytes (default 0 meaning not checked). For every node and its
      # descendants, asserts each metric with a positive threshold is at least that threshold.
      for n in nodes:
        if mm > 0:
          self.assertGreaterEqual(n.exec_micros, mm)
        if mam > 0:
          self.assertGreaterEqual(n.accelerator_exec_micros, mam)
        if mcm > 0:
          self.assertGreaterEqual(n.cpu_exec_micros, mcm)
        if mb > 0:
          self.assertGreaterEqual(n.requested_bytes, mb)
        if mpb > 0:
          self.assertGreaterEqual(n.peak_bytes, mpb)
        if mrb > 0:
          self.assertGreaterEqual(n.residual_bytes, mrb)
        if mob > 0:
          self.assertGreaterEqual(n.output_bytes, mob)
        check_min(n.children, mm, mam, mcm, mb, mpb, mrb, mob)

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildSmallModel()
      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      min_val = random.randint(0, 10000)

      opts = builder(builder.time_and_memory(
          min_micros=min_val)).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mm=min_val)

      opts = builder(builder.time_and_memory(
          min_accelerator_micros=min_val)).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mam=min_val)

      opts = builder(builder.time_and_memory(
          min_cpu_micros=min_val)).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mcm=min_val)

      opts = builder(builder.time_and_memory(
          min_bytes=min_val)).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mb=min_val)

      opts = builder(builder.time_and_memory(
          min_peak_bytes=min_val)).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mpb=min_val)

      opts = builder(builder.time_and_memory(
          min_residual_bytes=min_val)).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mrb=min_val)

      opts = builder(builder.time_and_memory(
          min_output_bytes=min_val)).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mob=min_val)

  def testSelectOption(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the last check lists the not-selected name with underscores (requested_bytes)
    # while the report uses spaces, so that negative check can never fail.
    # Scenario: profile the small model three times with different selected attribute groups
    # (time, bytes, then peak/residual/output bytes) writing to a file, and after each assert
    # the report contains the expected column titles and omits the ones not selected.
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')

    def check_selection(selected, not_selected):
      # [FUNCTION DOC]
      # HELPER: Assertion helper for the select option test.
      # Inputs: selected, list of strings that must appear in the dumped report file;
      # not_selected, list of strings that must not appear. Reads the output file and asserts
      # each selected string is found (at a position after the start) and each other is absent.
      with gfile.Open(outfile, 'r') as f:
        s = f.read()
        for attr in selected:
          self.assertTrue(s.find(attr) > 0, s)
        for attr in not_selected:
          self.assertFalse(s.find(attr) > 0, s)

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = lib.BuildSmallModel()
      self.evaluate(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      opts = builder(
          builder.time_and_memory()).with_file_output(outfile).select(
              ['micros']).build()
      _ = model_analyzer.profile(sess.graph, run_meta=run_meta, options=opts)
      check_selection(['total execution time', 'accelerator execution time'],
                      ['bytes'])

      opts = builder(
          builder.time_and_memory()).with_file_output(outfile).select(
              ['bytes']).build()
      _ = model_analyzer.profile(sess.graph, run_meta=run_meta, options=opts)
      check_selection(['requested bytes'],
                      ['peak bytes', 'residual bytes', 'output bytes'])

      opts = builder(
          builder.time_and_memory()).with_file_output(outfile).select(
              ['peak_bytes', 'residual_bytes', 'output_bytes']).build()
      _ = model_analyzer.profile(sess.graph, run_meta=run_meta, options=opts)
      check_selection(['peak bytes', 'residual bytes', 'output bytes'],
                      ['requested_bytes'])

  def _trainLoop(self, train_op, train_steps, time_dir, time_step, memory_dir,
                 memory_step, profile_dir, dump_step):
    # [FUNCTION DOC]
    # HELPER: Training loop driver for the automatic profiling test.
    # Inputs: train_op, the operation to evaluate each step; train_steps, number of steps to run;
    # time_dir and memory_dir, directories receiving automatic time and memory reports;
    # time_step and memory_step, lists of steps at which those reports are expected; profile_dir,
    # directory receiving dumped profiles; dump_step, list of steps at which a profile dump named
    # after the step is expected. Initializes variables, then each step evaluates the op and
    # asserts exactly one file exists in the time (memory) directory with the expected content
    # on scheduled steps (and removes it) or none otherwise, and that the profile directory
    # contains the expected dump on dump steps (removing it) or does not exist yet / is empty.
    with session.Session(config=self._no_rewrite_session_config()) as sess:
      self.evaluate(variables.global_variables_initializer())
      # start from 1 because variable_initializer took one step.
      for i in range(1, train_steps + 1):
        _ = self.evaluate(train_op)
        if i in time_step:
          ret = gfile.ListDirectory(time_dir)
          self.assertEqual(len(ret), 1)
          self.assertTrue(
              gfile.Open(os.path.join(time_dir, ret[0]), 'r').read().find(
                  'execution time') > 0)
          _ = [gfile.Remove(os.path.join(time_dir, x)) for x in ret]
        else:
          self.assertEqual(len(gfile.ListDirectory(time_dir)), 0)
        if i in memory_step:
          ret = gfile.ListDirectory(memory_dir)
          self.assertEqual(len(ret), 1)
          self.assertTrue(
              gfile.Open(os.path.join(memory_dir, ret[0]), 'r').read().find(
                  'requested bytes') > 0)
          _ = [gfile.Remove(os.path.join(memory_dir, x)) for x in ret]
        else:
          self.assertEqual(len(gfile.ListDirectory(memory_dir)), 0)
        if i in dump_step:
          ret = gfile.ListDirectory(profile_dir)
          self.assertAllEqual(ret, ['profile_%d' % i])
          _ = [gfile.Remove(os.path.join(profile_dir, x)) for x in ret]
        else:
          if i < dump_step[0]:
            self.assertFalse(gfile.Exists(profile_dir))
          else:
            self.assertEqual(len(gfile.ListDirectory(profile_dir)), 0)

  @test_util.run_v1_only('b/120545219')
  def testAutoProfiling(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): within a profile context tracing steps one to three and dumping
    # at steps three and four, register automatic scope profiling of time at steps two and three
    # and memory at steps one and three, with reports written to separate directories. Runs ten
    # training steps through the loop helper, which asserts the files appear only on the
    # scheduled steps.
    ops.reset_default_graph()
    time_dir = os.path.join(test.get_temp_dir(), 'time')
    memory_dir = os.path.join(test.get_temp_dir(), 'memory')
    profile_dir = os.path.join(test.get_temp_dir(), 'dir/dir2/profile')
    # TODO(xpan): Should we create parent directory for them?
    gfile.MkDir(time_dir)
    gfile.MkDir(memory_dir)

    time_opts = (
        builder(builder.time_and_memory()).with_file_output(
            os.path.join(time_dir, 'profile')).select(['micros']).build())
    memory_opts = (
        builder(builder.time_and_memory()).with_file_output(
            os.path.join(memory_dir, 'profile')).select(['bytes']).build())

    time_steps = [2, 3]
    memory_steps = [1, 3]
    dump_steps = [3, 4]

    x = lib.BuildSmallModel()
    with profile_context.ProfileContext(
        profile_dir, trace_steps=[1, 2, 3], dump_steps=[3, 4]) as pctx:
      pctx.add_auto_profiling('scope', time_opts, time_steps)
      pctx.add_auto_profiling('scope', memory_opts, memory_steps)

      self._trainLoop(x, 10, time_dir, time_steps, memory_dir, memory_steps,
                      profile_dir, dump_steps)

  @test_util.run_v1_only('b/120545219')
  def testOOM(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if running the oversized multiplication does not raise, the test passes
    # without asserting anything; it also does nothing without a GPU.
    # Scenario (graph mode, GPU only): multiply two huge random tensors on the GPU with tensor
    # allocation reporting upon out-of-memory enabled. When the expected exception occurs,
    # asserts its text mentions the failed allocation shape and reports the allocation sizes
    # (gibibytes and mebibytes) from both random operations.
    if not test.is_gpu_available():
      return
    ops.reset_default_graph()
    with ops.device('/device:GPU:0'):
      a = random_ops.random_normal([1, 10000, 20000], name='test_random1')
      b = random_ops.random_normal([30000, 10000, 1], name='test_random2')
      c = a * b

    try:
      with session.Session(config=self._no_rewrite_session_config()) as sess:
        sess.run(
            c,
            options=config_pb2.RunOptions(
                report_tensor_allocations_upon_oom=True))
    except Exception as e:  # pylint: disable=broad-except
      exception_str = '%s' % e
      # This trace reports allocations for to random tensor.
      self.assertTrue('OOM when allocating tensor with shape[30000,10000,20000]'
                      in exception_str)
      mat = re.search('(.*)GiB from test_random2/RandomStandardNormal',
                      exception_str)
      self.assertGreater(float(mat.group(1)), 0.0)
      mat = re.search('(.*)MiB from test_random1/RandomStandardNormal',
                      exception_str)
      self.assertGreater(float(mat.group(1)), 0.0)

  @test_util.run_v1_only('b/120545219')
  def testDistributedOOM(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: passes silently when no exception is raised, and does nothing without a GPU.
    # Scenario (graph mode, GPU only): create a local two-worker cluster, place one huge random
    # tensor on each worker and multiply them on worker one with allocation reporting on
    # out-of-memory. Asserts the error text reports usage for worker one's device and allocator,
    # includes the second random tensor's size, and omits the first tensor, which lives on the
    # other worker.
    if not test.is_gpu_available():
      return
    ops.reset_default_graph()

    workers, _ = test_util.create_local_cluster(2, 0)

    with ops.device('/job:worker/replica:0/task:0/gpu:0'):
      a = random_ops.random_normal([1, 10000, 20000], name='test_random1')
    with ops.device('/job:worker/replica:0/task:1/gpu:0'):
      b = random_ops.random_normal([30000, 10000, 1], name='test_random2')
      c = a * b

    try:
      with session.Session(workers[1].target) as sess:
        sess.run(
            c,
            options=config_pb2.RunOptions(
                report_tensor_allocations_upon_oom=True))
    except Exception as e:  # pylint: disable=broad-except
      exception_str = '%s' % e
      # test_random2 is reported because it's allocated in worker 1.
      self.assertTrue('Current usage from device: '
                      '/job:worker/replica:0/task:1/device:GPU:0, '
                      'allocator: GPU_0_bfc' in exception_str)
      mat = re.search('(.*)GiB from test_random2/RandomStandardNormal',
                      exception_str)
      self.assertGreater(float(mat.group(1)), 0.0)
      # test_random1 is not reported because it's allocated in worker 0.
      mat = re.search('(.*)MiB from test_random1/RandomStandardNormal',
                      exception_str)
      self.assertTrue(mat is None)

  @test_util.run_v1_only('b/120545219')
  def testTrackPersistentBytes(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): multiply two constants with graph-size rewrite threshold
    # disabled, run twice with full tracing and profile each run by scope with bytes, peak,
    # output and residual bytes selected. Asserts the multiplication node has positive peak,
    # output and residual bytes and that those values are identical across the two runs.
    ops.reset_default_graph()
    a = array_ops.constant(np.ones((100, 100)))
    b = array_ops.constant(np.ones((100, 100)))
    c = a * b
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.min_graph_nodes = -1

    with session.Session(config=config) as sess:
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      sess.run(c, options=run_options, run_metadata=run_metadata)

      options = option_builder.ProfileOptionBuilder.time_and_memory()
      options['min_bytes'] = 0
      options['select'] = ('bytes', 'peak_bytes', 'output_bytes',
                           'residual_bytes')
      ret = model_analyzer.profile(
          sess.graph, run_meta=run_metadata, cmd='scope', options=options)

      run_metadata = config_pb2.RunMetadata()
      sess.run(c, options=run_options, run_metadata=run_metadata)
      ret2 = model_analyzer.profile(
          sess.graph, run_meta=run_metadata, cmd='scope', options=options)

      n = lib.SearchTFProfNode(ret, 'mul')
      n2 = lib.SearchTFProfNode(ret2, 'mul')
      self.assertGreater(n.peak_bytes, 0)
      self.assertGreater(n.output_bytes, 0)
      self.assertGreater(n.residual_bytes, 0)
      self.assertEqual(n.peak_bytes, n2.peak_bytes)
      self.assertEqual(n.output_bytes, n2.output_bytes)
      self.assertEqual(n.residual_bytes, n2.residual_bytes)

  @test_util.run_v1_only('b/120545219')
  def testTraceLoopBytes(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, GPU only; returns immediately otherwise): build a while loop on the
    # GPU that repeatedly squares a matrix for 100 iterations and compute its gradient. Run with
    # full tracing and profile by scope with no output. Asserts the total requested bytes exceed
    # one million, showing loop iterations' memory is accounted.
    if not test.is_gpu_available():
      return
    ops.reset_default_graph()
    steps = 100

    with ops.device('/gpu:0'):
      x = array_ops.ones((100, 100), dtype=dtypes.float32)
      n = array_ops.constant(steps, dtype=dtypes.int32)
      x1 = array_ops.ones((100, 100))

      x *= x1

      def loop_body(i, x):
        # [FUNCTION DOC]
        # HELPER: Loop body for the traced while loop. Inputs: i, iteration counter; x, matrix.
        # Returns the incremented counter and the element-wise square of the matrix.
        x *= x
        return i + 1, x

      _, y = while_loop.while_loop(lambda i, x: i < n, loop_body,
                                   [array_ops.constant(0), x])

    grad = gradients.gradients(y, [x1])

    with session.Session(config=self._no_rewrite_session_config()) as sess:
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      sess.run(grad, options=run_options, run_metadata=run_metadata)

      options = option_builder.ProfileOptionBuilder.time_and_memory()
      options['min_bytes'] = 0
      options['min_micros'] = 0
      options['select'] = ('bytes', 'peak_bytes', 'output_bytes',
                           'residual_bytes')
      options['output'] = 'none'
      ret_pb = model_analyzer.profile(
          sess.graph, run_meta=run_metadata, cmd='scope', options=options)
      self.assertGreater(ret_pb.total_requested_bytes, 1000000)


if __name__ == '__main__':
  test.main()
