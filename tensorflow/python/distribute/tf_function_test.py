# [MODULE DOC]
# Tests for tf.function combined with distribution strategies: device placement in functions
# under a strategy scope, variable reads across devices, and retracing behavior around saving
# contexts. At run time enables TF2 behavior and starts the standard test main.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.function + distribution strategies."""

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options

FLAGS = flags.FLAGS


class TFFunctionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized eager-mode tests of how tf.function interacts with distribution strategies:
  # default device placement inside functions under a strategy scope, reading distributed
  # variables from within functions, and retracing when a function is first traced inside or
  # outside a saving context. Each test receives a strategy (named distribution).

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup and then turns off the global option that runs functions eagerly so
    # every test starts from the default (graph-tracing) behavior.
    super().setUp()
    # Clear the state for every test.
    def_function.run_functions_eagerly(False)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          run_functions_eagerly=[True, False]
      ))
  def testDefaultDeviceInsideFunctionWithScope(
      self, distribution, run_functions_eagerly):
    # [FUNCTION DOC]
    # Scenario: with the run-functions-eagerly option set to the parameter value, enters the
    # strategy scope and a CPU device scope, then calls a traced function that creates a scalar
    # and asserts on its device. Expected device: the canonicalized CPU device on the first worker
    # (or no worker if the strategy has none) when functions run eagerly, otherwise an empty
    # device string (no placement in a graph).

    def_function.run_functions_eagerly(run_functions_eagerly)
    try:
      worker = distribution.extended.worker_devices[0]
    except RuntimeError:
      worker = None
    expected_device = (device_util.canonicalize("cpu:0", worker)
                       if run_functions_eagerly else "")
    with distribution.scope():
      with ops.device_v2("cpu:0"):
        @def_function.function
        def add():
          # [FUNCTION DOC]
          # Traced function: creates a scalar one, asserts its device equals the expected device,
          # and returns it plus one.
          one = array_ops.ones([])
          self.assertEqual(expected_device, one.device)
          return one + 1

        add()

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          run_functions_eagerly=[True, False]
      ))
  def testDefaultDeviceInsideNestedFunctionWithScope(
      self, distribution, run_functions_eagerly):
    # [FUNCTION DOC]
    # Scenario: like the previous test but with nesting: inside the strategy scope an outer
    # traced function enters a CPU device scope and calls an inner traced function that creates a
    # scalar and asserts on its device. Expected: canonicalized CPU device on the first worker
    # when running functions eagerly, otherwise an empty device string.

    def_function.run_functions_eagerly(run_functions_eagerly)
    try:
      worker = distribution.extended.worker_devices[0]
    except RuntimeError:
      worker = None
    expected_device = (device_util.canonicalize("cpu:0", worker)
                       if run_functions_eagerly else "")
    with distribution.scope():
      @def_function.function
      def foo():
        # [FUNCTION DOC]
        # Outer traced function: enters a CPU device scope, defines the inner traced function and
        # calls it. Returns nothing.
        with ops.device("cpu:0"):

          @def_function.function
          def bar():
            # [FUNCTION DOC]
            # Inner traced function: creates a scalar one, asserts its device equals the expected
            # device, and returns it plus one.
            one = array_ops.ones([])
            self.assertEqual(expected_device, one.device)
            return one + 1

          bar()

      foo()

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          run_functions_eagerly=[True, False]
      ))
  def testReadVariableInsideFunction(self, distribution, run_functions_eagerly):
    # [FUNCTION DOC]
    # Scenario: skipped when functions are not run eagerly on TPU with the TFRT flag (no support
    # for the XLA launch op). Sets the eager option, determines the strategy parameter devices
    # (falling back to one CPU when the strategy does not define them), and creates a scalar
    # variable in the strategy scope. If it is a distributed variable, each component variable is
    # assigned its own index as a value so devices differ. A traced function reads the variable;
    # within the strategy scope and each device scope in turn, the read value must equal that
    # device's index (checks retracing per device).
    if not run_functions_eagerly and config.list_physical_devices(
        "TPU") and FLAGS.tpu_use_tfrt:
      self.skipTest("TFRT does not support XlaLocalLaunch, see b/194517185")

    def_function.run_functions_eagerly(run_functions_eagerly)

    # Get devices on which variables will be placed. Default strategy does not
    # define this, so assume cpu:0 in that case.
    try:
      devices = distribution.extended.parameter_devices
    except RuntimeError:
      devices = ["cpu:0"]

    with distribution.scope():
      v = variables.Variable(0.)
      if isinstance(v, values.DistributedVariable):
        for i in range(len(devices)):
          # NOTE: Assigning manually to component variables so we can test
          # different values on different devices. Using .assign on the
          # mirrored variable itself will lead to a synchronization which
          # will prohibit testing different values.
          replica_variable = v._values[i]
          replica_variable.assign(math_ops.cast(i, dtypes.float32))

    @def_function.function
    def read():
      # [FUNCTION DOC]
      # Traced function returning the current value of the variable under test.
      return v.read_value()

    # Verify that the value from each device is read, when in that device
    # scope. Doing this inside strategy scope is needed to force function
    # retracing on each device, otherwise `read()` will only be traced once
    # on the first device and following variable read will always read the value
    # on the first replica.
    with distribution.scope():
      for i, d in enumerate(devices):
        with ops.device(d):
          self.assertEqual(math_ops.cast(i, dtypes.float32), read())

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies, mode=["eager"]))
  def testRetraceOnSavingFirstTraceInScope(self, distribution):
    # [FUNCTION DOC]
    # Scenario: a variable is created in the strategy scope; a traced function that counts its
    # traces and adds one to the variable is first run through the strategy. Then it is called
    # inside a saving context: expected one extra trace. Calling it again inside a saving context
    # must not trace again.
    with distribution.scope():
      v = variables.Variable(0.)

    tracing_count = [0]

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Traced function: increments the trace counter (a Python side effect that only happens
      # while tracing) and returns the variable plus one.
      tracing_count[0] += 1
      return v + 1.

    distribution.run(func)
    prev_tracing_count = tracing_count[0]
    with save_context.save_context(save_options.SaveOptions()):
      func()
    self.assertEqual(prev_tracing_count + 1, tracing_count[0])

    prev_tracing_count = tracing_count[0]
    with save_context.save_context(save_options.SaveOptions()):
      func()
    self.assertEqual(prev_tracing_count, tracing_count[0])

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies, mode=["eager"]))
  def testRetraceOnSavingFirstTraceOutsideScope(self, distribution):
    # [FUNCTION DOC]
    # Scenario: same as the in-scope variant, but the first call of the traced function is made
    # directly, outside the strategy run. Calling it inside a saving context must trigger exactly
    # one additional trace, and a second call in a saving context no further trace.
    with distribution.scope():
      v = variables.Variable(0.)

    tracing_count = [0]

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Traced function: increments the trace counter (a Python side effect that only happens
      # while tracing) and returns the variable plus one.
      tracing_count[0] += 1
      return v + 1.

    func()
    prev_tracing_count = tracing_count[0]
    with save_context.save_context(save_options.SaveOptions()):
      func()
    self.assertEqual(prev_tracing_count + 1, tracing_count[0])

    prev_tracing_count = tracing_count[0]
    with save_context.save_context(save_options.SaveOptions()):
      func()
    self.assertEqual(prev_tracing_count, tracing_count[0])


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
