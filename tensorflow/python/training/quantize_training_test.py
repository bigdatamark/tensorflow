# [MODULE DOC]
# Unit tests for the quantize-training graph rewriter Python interface. Checks that the rewrite
# adds nodes to a simple multiplication graph and that a saver can save and restore the extra
# moving-average min/max variables the rewrite creates. Test module only; no import-time effects
# beyond imports, and runs through the test main entry point when executed directly.
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
"""Tests for the SWIG-wrapped quantize training rewriting."""

import os

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import quantize_training
from tensorflow.python.training import saver as saver_module


class PywrapQuantizeTrainingTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the quantize-training rewriter wrapper: basic rewrite and save/restore of
  # the quantization variables. No shared fixtures; each test builds its own graph and session.

  # Mainly to verify the python interface is working.
  # More tests for this function can be found in the related c++ tests.
  def testQuantizeTraining(self):
    # [FUNCTION DOC]
    # Scenario: in a session, build a graph multiplying two 1x1 constants (six and seven) and check
    # it evaluates to forty-two with exactly three nodes. Action: run the quantize-training rewrite
    # with eight bits on the session's graph definition. Expected: the result has more than three
    # nodes, showing a rewrite happened.
    with session.Session() as sess:
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')

      self.assertEqual(c.eval(), 42.0)
      self.assertEqual(len(sess.graph_def.node), 3)

      result = quantize_training.do_quantize_training_on_graphdef(
          sess.graph_def, 8)

      # We just want to guarantee that some rewrite happened.
      self.assertGreater(len(result.node), 3)

  # Test that save/restoring works for EMA variables generated in the
  # quantized training rewrite.
  @test_util.run_v1_only('The API is only expect to work with v1 session mode.')
  def testQuantizedSaveRestore(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): a graph with a constant a, a variable b and their matrix product.
    # The graph is rewritten for quantization with eight bits. The rewritten graph is imported into
    # a fresh graph, the variable initializer and product are run, and a saver saves to a temp file.
    # A third fresh graph imports the rewritten graph, restores the checkpoint, and the test asserts
    # b equals seven and the min/max quantization variables for a (six) and for b's read (seven)
    # were restored too. Side effect: writes a checkpoint under the test temp directory.
    save_path = os.path.join(self.get_temp_dir(), 'quantized_save_restore')

    g = ops.Graph()
    with session.Session(graph=g) as sess:
      a = constant_op.constant(6.0, shape=[1, 1], name='a')
      b = variable_v1.VariableV1(
          constant_op.constant(7.0, shape=[1, 1]), name='b')
      c = math_ops.matmul(a, b, name='matmul')

      init_op = variables.global_variables_initializer()

      saver = saver_module.Saver({'b': b})

      result = quantize_training.do_quantize_training_on_graphdef(
          sess.graph_def, 8)

    with ops.Graph().as_default() as g, session.Session(graph=g) as sess:
      _ = importer.import_graph_def(result, name='')

      # Initialize the variable.
      self.evaluate(g.get_operation_by_name(init_op.name))

      # Run the graph for one step to assign values to the quantization min/max
      # variables.
      self.evaluate(g.get_tensor_by_name(c.name))

      saver.save(sess, save_path)

    with ops.Graph().as_default() as g, session.Session(graph=g) as sess:
      _ = importer.import_graph_def(result, name='')

      # When we restore the saved variabled, the quantization variables should
      # be restored as well.
      saver.restore(sess, save_path)
      self.assertEqual(7.0, sess.run(g.get_tensor_by_name('b:0')))
      self.assertEqual(6.0, sess.run(g.get_tensor_by_name('a/Min/Variable:0')))
      self.assertEqual(6.0, sess.run(g.get_tensor_by_name('a/Max/Variable:0')))
      self.assertEqual(7.0,
                       sess.run(g.get_tensor_by_name('b/read/Min/Variable:0')))
      self.assertEqual(7.0,
                       sess.run(g.get_tensor_by_name('b/read/Max/Variable:0')))


if __name__ == '__main__':
  test.main()
