# [MODULE DOC]
# Tests for the per-replica distributed value container from the values library: type spec,
# component round trip, nest handling, graph tensors, function tracing behavior and conditionals.
# Run through the distribute test main entry point.

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the distributed values library."""

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import test_util as ds_test_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.util import nest


class PerReplicaTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the per-replica value container: its type spec and component round trip, nest
  # flattening, graph tensor behavior, interaction with traced functions (no retracing, returning
  # it), and use as branches of a conditional. No shared fixtures.

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testTypeSpec(self):
    # [FUNCTION DOC]
    # Scenario (eager): wrap one scalar float tensor in a per-replica value.
    # Expected: its type spec's value specs are a tuple holding one scalar float32 tensor spec.
    vals = (constant_op.constant(1.),)
    per_replica = values_lib.PerReplica(vals)

    spec = per_replica._type_spec
    self.assertEqual(spec._value_specs,
                     (tensor_spec.TensorSpec([], dtypes.float32),))

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testTypeSpecRoundTrip(self):
    # [FUNCTION DOC]
    # Scenario (eager): convert a per-replica value to components with its type spec and rebuild
    # it from those components. Expected: the rebuilt values equal the original values.
    vals = (constant_op.constant(1.),)
    per_replica = values_lib.PerReplica(vals)

    spec = per_replica._type_spec
    tensor_list = spec._to_components(per_replica)
    reconstructed = spec._from_components(tensor_list)

    self.assertAllEqual(per_replica.values, reconstructed.values)

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testTypeSpecNest(self):
    # [FUNCTION DOC]
    # Scenario (eager): per-replica value with a scalar and a two-element vector; map a function
    # adding ten over it with composite expansion (exercising flatten and pack).
    # Expected: the result is a per-replica value with two entries: 11 and the vector 15, 16.
    vals = (constant_op.constant(1.), constant_op.constant([5., 6.0]),)
    per_replica = values_lib.PerReplica(vals)

    # Note: nest.map_structure exercises nest.flatten and
    # nest.pack_sequence_as.
    result = nest.map_structure(
        lambda t: t + 10, per_replica, expand_composites=True)

    self.assertLen(result.values, 2)
    self.assertAllEqual(result.values[0], 11.)
    self.assertAllEqual(result.values[1], [15., 16.0])

  @test_util.run_in_graph_and_eager_modes
  def testIsGraphTensor(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): flatten a per-replica value with composite expansion.
    # Expected: each component has a graph attribute exactly when not executing eagerly.
    per_replica = values_lib.PerReplica((constant_op.constant(1.),))
    for t in nest.flatten(per_replica, expand_composites=True):
      self.assertEqual(hasattr(t, "graph"), not context.executing_eagerly())

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testDoesNotTriggerFunctionTracing(self):
    # [FUNCTION DOC]
    # Scenario (eager): a traced identity function records each trace. Call it once with a
    # per-replica value, then five more times with values rebuilt from doubled components.
    # Expected: the first call traces; later calls return per-replica values equal to the inputs
    # and never retrace.
    traces = []

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: identity function that appends to the trace list whenever it is traced, and
      # returns its input.
      traces.append(None)  # Only happens on trace.
      return x

    per_replica = values_lib.PerReplica((constant_op.constant(1.),))

    # Trace once.
    f(per_replica)
    self.assertNotEmpty(traces)
    del traces[:]

    per_replica_spec = per_replica._type_spec
    for _ in range(5):
      vals = per_replica_spec._to_components(per_replica)
      vals = [v * 2 for v in vals]
      per_replica = per_replica_spec._from_components(vals)

      output = f(per_replica)
      self.assertIsInstance(output, values_lib.PerReplica)
      self.assertAllEqual(output._values, per_replica._values)
      self.assertEmpty(traces)  # Make sure we're not re-tracing `f`.

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testFunctionCanReturnPerReplica(self):
    # [FUNCTION DOC]
    # Scenario (eager): pass a per-replica value through an identity traced function.
    # Expected: the output is a different object whose components equal the input's, with an
    # equal type spec.
    f = def_function.function(lambda x: x)
    x = values_lib.PerReplica((constant_op.constant(1.),))
    y = f(x)
    self.assertIsNot(x, y)
    nest.map_structure(self.assertAllEqual, x, y, expand_composites=True)
    self.assertEqual(x._type_spec, y._type_spec)

  @test_util.run_in_graph_and_eager_modes
  def testCondWithTensorValues(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): conditional with two per-replica branches holding string
    # tensors of different shapes, condition true via a placeholder with default.
    # Expected: result has one value equal to "a" (the first branch).
    per_replica_1 = values_lib.PerReplica((constant_op.constant("a"),))
    per_replica_2 = values_lib.PerReplica((constant_op.constant(["b", "c"]),))
    condition = array_ops.placeholder_with_default(True, [])

    result = cond.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.run_in_graph_and_eager_modes
  def testCondWithValuesConvertibleToTensor(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): conditional with two per-replica branches holding plain
    # strings (convertible to tensors), condition true.
    # Expected: result has one value equal to "a".
    per_replica_1 = values_lib.PerReplica(("a",))
    per_replica_2 = values_lib.PerReplica(("b",))
    condition = array_ops.placeholder_with_default(True, [])

    result = cond.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.build_as_function_and_v1_graph
  def testCondWithValuesNotConvertibleToTensor(self):
    # [FUNCTION DOC]
    # Scenario (function-building and v1 graph modes): conditional with branches whose per-replica
    # values are sets, which cannot be turned into tensors.
    # Expected: raises a type error saying a type spec could not be built.
    per_replica_1 = values_lib.PerReplica(({"a"},))
    per_replica_2 = values_lib.PerReplica(({"b", "c"},))
    condition = array_ops.placeholder(dtypes.bool, [])

    with self.assertRaisesRegex(TypeError, "Could not build a TypeSpec for"):
      cond.cond(
          condition, lambda: per_replica_1, lambda: per_replica_2)

if __name__ == "__main__":
  ds_test_util.main()
