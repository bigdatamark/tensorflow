# [MODULE DOC]
# Test module checking that templates created with make_template work with MirroredStrategy,
# in particular when the template function contains a cross-replica merge call between variable
# creations. Defines one test class; run through the standard test main entry point. No
# import-time side effects.
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
"""Tests for make_template used with MirroredStrategy."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class TemplateMirroredStrategyTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the interaction of variable-scope templates (make_template) with
  # MirroredStrategy. Holds no shared fixtures; contains a single test.

  @test_util.disable_tfrt("Strategy not supported yet.")
  def test_merge_call(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, a template wraps a function that creates one variable, performs
    # a merge call across replicas, then creates a second variable and returns the product of the
    # two. The test is skipped when no GPU is available. Setup: a mirrored strategy over one CPU
    # and one GPU device runs the template and gathers per-replica results. Action: initialize
    # global variables and evaluate the outputs. Expected: both replicas yield forty-two
    # (twenty-one times two). Disabled under the TFRT runtime.
    with ops.Graph().as_default():
      # The test is testing a v1 only function.
      if not test.is_gpu_available():
        self.skipTest("No GPU available")

      def fn():
        # [FUNCTION DOC]
        # Template body run on each replica: creates the variables with fixed initial values
        # twenty-one and two (a merge call with a no-op function sits between them) and returns
        # their product.
        var1 = variable_scope.get_variable(
            "var1", shape=[], initializer=init_ops.constant_initializer(21.))
        distribute_lib.get_replica_context().merge_call(lambda _: ())
        var2 = variable_scope.get_variable(
            "var2", shape=[], initializer=init_ops.constant_initializer(2.))
        return var1 * var2

      temp = template.make_template("my_template", fn)

      strategy = mirrored_strategy.MirroredStrategy(["/cpu:0", "/gpu:0"])
      out = strategy.experimental_local_results(
          strategy.run(temp))

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual([42., 42.], self.evaluate(out))


if __name__ == "__main__":
  test.main()
