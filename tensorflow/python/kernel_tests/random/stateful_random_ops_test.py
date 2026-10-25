# [MODULE DOC]
# Unit tests for the stateful random number generator API in the stateful random ops module
# (imported here under the short name random): the Generator class, its construction from state,
# seed, key and counter or non-deterministic state, algorithms selected by name, integer or enum,
# seeding, splitting, skipping, resetting, variable-backed state, copying, checkpoint save and
# restore, use inside tf.function (including XLA and tf.function arguments), the global generator,
# agreement with the older stateless-seed random ops on CPU and GPU, value ranges and statistical
# distribution checks, and error messages from the raw generated ops and from determinism mode.
# Module level state: two global generator holders used by tests to create variables only once
# inside traced functions, plus lists of float and integer dtypes used for parameterization.
# When run as a script it disables soft device placement and starts the standard test main.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stateful_random_ops.py."""

import os
import re

from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util as random_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import stateful_random_ops as random
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


g_seeded = None
g_unseeded = None

FLOATS = [dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64]
INTS = [dtypes.int32, dtypes.int64]


class StatefulRandomOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized TensorFlow test case for the stateful random generator. Most tests run only in
  # v2 (eager) mode; a few need CUDA or XLA devices and skip themselves otherwise. Shared fixture:
  # setUp splits the first physical CPU into two logical devices. Provides small assertion helpers
  # (all-different check and a regex search that takes the pattern first). Tests mutate the
  # process-wide global generator and module level generator holders.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture run before every test. Calls the parent setUp, then configures the first physical
    # CPU device to be exposed as two logical CPU devices (so device placement tests have more than
    # one CPU). Returns nothing; changes the global device configuration of the runtime.
    super(StatefulRandomOpsTest, self).setUp()
    physical_devices = config.list_physical_devices("CPU")
    config.set_logical_device_configuration(
        physical_devices[0], [
            context.LogicalDeviceConfiguration(),
            context.LogicalDeviceConfiguration()
        ])

  def testCreateRNGStateIntSeed(self):
    # [FUNCTION DOC]
    # Scenario: create_rng_state is given a very large integer seed (more than 64 bits, with high
    # bits set to check overflow tolerance) and the Philox algorithm. Asserts the state equals the
    # low 64 bits and next 64 bits of the seed, each converted from unsigned to signed integer,
    # followed by zeros up to the Philox state size.
    """Tests `create_rng_state` when `seed` is int."""
    # using leading 'F' to test overflow tolerance
    state = random.create_rng_state(0xFFFF222233334444FFAA666677778888,
                                    random.RNG_ALG_PHILOX)
    self.assertAllEqual(
        list(map(random._uint_to_int,
                 [0xFFAA666677778888, 0xFFFF222233334444] +
                 [0] * (random.PHILOX_STATE_SIZE - 2))),
        state)

  def assertAllDifferent(self, tensors):
    # [FUNCTION DOC]
    # HELPER: custom assertion shared by the tests in this class.
    # Input: a list of tensors that may have different shapes. Flattens each, concatenates them,
    # converts to a plain list (requires eager tensors) and asserts no value occurs twice.
    # Raises an assertion failure on any duplicate; returns nothing.
    """Checks that there are no duplicate elements anywhere among the tensors.

    Args:
      tensors: a list of tensors. They can have different shapes.
    """
    tensors = [array_ops.reshape(t, shape=[-1]) for t in tensors]
    ls = array_ops.concat(tensors, axis=0).numpy().tolist()
    self.assertAllEqual(len(ls), len(set(ls)))

  @test_util.run_v2_only
  def testNonDeterministicInts(self):
    # [FUNCTION DOC]
    # Scenario: calls non_deterministic_ints twice with a 2 by 3 shape and int64 dtype. Asserts the
    # first result has the requested shape and dtype and that all values across the two results are
    # distinct (probabilistic, tiny flake chance). Runs in v2 only.
    """Tests that non_deterministic_ints returns different results every time.

    This test is flaky, but with very low probability of failing.
    """
    shape = [2, 3]
    dtype = dtypes.int64
    a = random.non_deterministic_ints(shape=shape, dtype=dtype)
    self.assertAllEqual(shape, a.shape)
    self.assertEqual(dtype, a.dtype)
    b = random.non_deterministic_ints(shape, dtype=dtype)
    self.assertAllDifferent([a, b])

  def assertRegex(self, pattern, text):
    # [FUNCTION DOC]
    # HELPER: assertion shared by the tests in this class.
    # KNOWN ISSUE: this overrides the base class assertRegex but with the argument order reversed
    # (pattern first, text second, whereas the base class takes text first); it also searches
    # rather than matches, and its failure message is custom.
    # Asserts that a regular-expression search of the pattern in the text finds a match; fails
    # with a message showing both otherwise. Returns nothing.
    self.assertTrue(
        re.search(pattern, text),
        "Can't find pattern '%s' in text '%s'" % (pattern, text))

  @parameterized.parameters(list(random_ops_util.Algorithm))
  @test_util.run_v2_only
  def testBatchSeeds(self, alg):
    # [FUNCTION DOC]
    # Scenario (parameterized over every RNG algorithm, v2 only): builds a generator from a seed and
    # returns early for ThreeFry since no CPU or GPU kernels exist for it. For other algorithms it
    # asserts that two calls producing int64 keys give different values, that two calls to
    # make_seeds give different seeds, that split returns the requested number of generators whose
    # full-range integer draws are all different, and that make_seeds also works three times inside
    # a traced function.
    """Test for batch seeds."""
    shape = [2, 3]
    count = 6
    gen = random.Generator.from_seed(1234, alg=alg)
    if alg == random_ops_util.Algorithm.THREEFRY:
      # We don't have CPU/GPU kernels for ThreeFry yet.
      return
    keys1 = gen._make_int64_keys(shape=shape)
    keys2 = gen._make_int64_keys(shape=shape)
    self.assertAllDifferent([keys1, keys2])
    seeds1 = gen.make_seeds(count=count)
    seeds2 = gen.make_seeds(count=count)
    self.assertAllDifferent([seeds1[0, :], seeds2[0, :]])
    gens = gen.split(count=count)
    self.assertAllEqual(count, len(gens))
    randoms = [g.uniform_full_int(shape=shape, dtype=dtypes.int32)
               for g in gens]
    self.assertAllDifferent(randoms)
    # Tests graph mode.
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function used to check make_seeds works in graph mode. Takes no arguments
      # and returns the seeds from asking the enclosing generator for the enclosing count of seeds.
      return gen.make_seeds(count=count)
    for _ in range(3):
      f()

  @parameterized.parameters(list(random_ops_util.Algorithm))
  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testCrossDeviceSplit(self, alg):
    # [FUNCTION DOC]
    # Scenario (parameterized over algorithms, v2 and CUDA only): a generator is created on CPU and
    # its state is on a CPU device; ThreeFry returns early (no kernels). Otherwise splitting into
    # ten generators while placed on the GPU must yield generators whose state lives on a GPU
    # device. Uses the regex helper to check device names.
    """Tests that a CPU RNG can split into RNGs on GPU."""
    with ops.device("/device:CPU:0"):
      gen = random.Generator.from_seed(1234, alg=alg)  # gen is on CPU
      self.assertRegex("CPU", gen.state.device)
    if alg == random_ops_util.Algorithm.THREEFRY:
      # We don't have CPU/GPU kernels for ThreeFry yet.
      return
    with ops.device(test_util.gpu_device_name()):
      gens = gen.split(count=10)  # gens are on GPU
      self.assertRegex("GPU", gens[0].state.device)

  @parameterized.parameters(list(random_ops_util.Algorithm))
  @test_util.run_v2_only
  def testSplitInFunction(self, alg):
    # [FUNCTION DOC]
    # Scenario (parameterized over algorithms, v2 only): splitting a generator inside a traced
    # function and drawing a normal sample from each of the two results must trace and run without
    # error. ThreeFry returns early (no kernels). Generators are stored in a one-element list so
    # they are created only on the first trace. Asserts nothing explicitly.
    g = random.Generator.from_seed(1, alg=alg)
    if alg == random_ops_util.Algorithm.THREEFRY:
      # We don't have CPU/GPU kernels for ThreeFry yet.
      return
    new_g = [None]  # using list as mutable cells
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function under test. On its first trace it splits the enclosing generator
      # into two and stores them in a mutable cell; always returns a list of two scalar normal
      # samples, one from each split generator. Takes no arguments.
      if new_g[0] is None:  # avoid creating variable in 2nd trace
        new_g[0] = g.split(2)
      return [new_g[0][i].normal([]) for i in range(2)]
    f()

  def testFnVars(self):
    # [FUNCTION DOC]
    # Scenario: a generator seeded with 0 is used in a traced function; the concrete function is
    # obtained and asserted to list the generator state variable among its variables.
    """Tests that RNG variable is added to ConcreteFunction.variables."""
    rng = random.Generator.from_seed(0)
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function whose only job is to use the generator. Takes no arguments;
      # returns a scalar normal sample from the enclosing generator.
      return rng.normal([])

    concrete = f.get_concrete_function()
    self.assertIn(rng.state, concrete.variables)

  @test_util.run_v2_only
  def testReset(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): iterates over three ways to reset a generator (from an explicit state,
    # from a seed, from a key and counter). After each reset a normal sample is taken as the
    # expectation; then a traced function performing the same reset and sampling is run twice and
    # must reproduce the expected sample each time.
    shape = [2, 3]
    gen = random.Generator.from_seed(0)
    for resetter in [
        lambda g: g.reset(state=[1, 2, 3]),
        lambda g: g.reset_from_seed(1234),
        lambda g: g.reset_from_key_counter(key=1, counter=[2, 3]),
    ]:
      resetter(gen)
      expected_normal = gen.normal(shape)
      @def_function.function
      def f(resetter):
        # [FUNCTION DOC]
        # HELPER: traced function under test. Input: a callback that resets the enclosing generator.
        # Applies the reset inside the traced function, then returns a normal sample of the shared
        # shape from that generator.
        resetter(gen)
        return gen.normal(shape)
      def check_results(expected_normal, v):
        # [FUNCTION DOC]
        # HELPER: asserts the actual value v is exactly equal to the expected normal sample. Returns
        # nothing.
        self.assertAllEqual(expected_normal, v)
      check_results(expected_normal, f(resetter))
      check_results(expected_normal, f(resetter))

  @test_util.run_v2_only
  def testGeneratorCreation(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): for three constructors (explicit state, seed, key and counter) a generator
    # is made eagerly and two normal samples are recorded. Then a traced function that constructs
    # the generator only on its first call (stored in a module level holder, which the test resets
    # to nothing first) is called twice; the outputs must equal the two recorded samples. Shows
    # generator creation inside a traced function behaves like variable creation.
    """Tests generator creation, in both eager and tf.function.

    The interaction between Generator creation and defun should be the same as
    tf.Variable.
    """
    shape = [2, 3]
    alg = random.RNG_ALG_PHILOX
    for constructor in [
        lambda: random.Generator(state=[1, 2, 3], alg=alg),
        lambda: random.Generator.from_seed(1234),
        lambda: random.Generator.from_key_counter(  # pylint: disable=g-long-lambda
            key=1, counter=[2, 3], alg=alg),
    ]:
      gen = constructor()
      # Tests tf.function
      expected_normal1 = gen.normal(shape)
      expected_normal2 = gen.normal(shape)
      global g_seeded
      g_seeded = None
      @def_function.function
      def f(constructor):
        # [FUNCTION DOC]
        # HELPER: traced function under test. Input: a callback that builds a generator. Declares
        # the module level generator holder global; only on the first trace builds the generator
        # and stores it there (so variables are created once). Returns a normal sample of the
        # shared shape from it.
        global g_seeded
        # defun'ed function should only create variables once
        if g_seeded is None:
          g_seeded = constructor()
        return g_seeded.normal(shape)
      def check_results(expected_normal, v):
        # [FUNCTION DOC]
        # HELPER: asserts the actual value v is exactly equal to the expected normal sample. Returns
        # nothing.
        self.assertAllEqual(expected_normal, v)
      check_results(expected_normal1, f(constructor))
      check_results(expected_normal2, f(constructor))

  @test_util.run_v2_only
  def testCreateGeneratorFromSymbolic(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): a traced function takes scalar and vector tensors as arguments and uses
    # them to build generators via seed, state and key and counter constructors, also exercising the
    # matching reset methods, then draws a scalar normal sample from each. The function is called
    # once with constant tensors; passes if it traces and runs without error.
    g = [None, None, None]  # using list as mutable cells
    @def_function.function
    def f(scalar, vector2, vector3):
      # [FUNCTION DOC]
      # HELPER: traced function under test. Inputs: a scalar tensor, a length-2 vector and a
      # length-3 vector. On the first trace builds three generators from them (seed, state and key
      # with counter constructors) and exercises the matching reset methods, storing them in a
      # mutable cell. Returns a list of three scalar normal samples, one per generator.
      if g[0] is None:  # avoid creating variable in 2nd trace
        g[0] = random.Generator.from_seed(scalar)
        g[0].reset_from_seed(scalar)  # also test reset
        g[1] = random.Generator.from_state(vector3, random.RNG_ALG_PHILOX)
        g[1].reset(vector3)
        g[2] = random.Generator.from_key_counter(
            scalar, vector2, random.RNG_ALG_PHILOX)
        g[2].reset_from_key_counter(scalar, vector2)
      return [g[i].normal([]) for i in range(3)]
    args = (1, [2, 2], [3, 3, 3])
    args = [constant_op.constant(v) for v in args]
    f(*args)

  @parameterized.parameters([
      (
          "philox",
          random_ops_util.Algorithm.PHILOX.value,
          random_ops_util.Algorithm.PHILOX,
      ),
      (
          "threefry",
          random_ops_util.Algorithm.THREEFRY.value,
          random_ops_util.Algorithm.THREEFRY,
      ),
      (
          "auto_select",
          random_ops_util.Algorithm.AUTO_SELECT.value,
          random_ops_util.Algorithm.AUTO_SELECT,
      ),
  ])
  @test_util.run_v2_only
  def testAlg(self, name, int_id, enum_id):
    # [FUNCTION DOC]
    # Scenario (parameterized over Philox, ThreeFry and auto-select; v2 only): generators created
    # with the same seed but the algorithm given by name, by integer id and by enum value must all
    # report the same algorithm. Except for ThreeFry (returns early, no kernels), their normal
    # samples must be identical.
    g_by_name = random.Generator.from_seed(1234, name)
    g_by_int = random.Generator.from_seed(1234, int_id)
    g_by_enum = random.Generator.from_seed(1234, enum_id)
    self.assertEqual(g_by_name.algorithm, g_by_int.algorithm)
    self.assertEqual(g_by_name.algorithm, g_by_enum.algorithm)
    if enum_id == random_ops_util.Algorithm.THREEFRY:
      # We don't have CPU/GPU kernels for ThreeFry yet.
      return
    shape = [3]
    output_by_name = g_by_name.normal(shape)
    output_by_int = g_by_int.normal(shape)
    output_by_enum = g_by_enum.normal(shape)
    self.assertAllEqual(output_by_name, output_by_int)
    self.assertAllEqual(output_by_name, output_by_enum)

  @test_util.run_v2_only
  def testGeneratorCreationWithVar(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): one generator is built from a list state and another from a variable
    # holding the same state. After each draws a normal sample, the two generators states must be
    # equal, showing the variable is used directly as state.
    """Tests creating generator with a variable.
    """
    alg = random.RNG_ALG_PHILOX
    state = [1, 2, 3]
    var = variables.Variable(state, dtype=random.STATE_TYPE)
    g = random.Generator(state=state, alg=alg)
    g_var = random.Generator(state=var, alg=alg)
    shape = [2, 3]
    g.normal(shape)
    g_var.normal(shape)
    self.assertAllEqual(g.state.read_value(), var.read_value())

  @test_util.run_v2_only
  def testGeneratorCreationUnseeded(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): a traced function creates a generator from non-deterministic state on its
    # first call (kept in a module level holder reset to nothing first) and returns a normal sample.
    # Asserts the output has the requested shape.
    """Tests generator creation, the unseeded case."""
    shape = [2, 3]
    global g_unseeded
    g_unseeded = None
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function under test. Takes no arguments. Declares the module level holder
      # global; on the first trace creates a generator from non-deterministic state. Returns a
      # normal sample of the enclosing shape.
      global g_unseeded
      # defun'ed function should only create variables once
      if g_unseeded is None:
        g_unseeded = random.Generator.from_non_deterministic_state()
      return g_unseeded.normal(shape)
    self.assertAllEqual(shape, f().shape)

  @test_util.run_v2_only
  def testGeneratorCopy(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): a generator is copied by passing it to the Generator constructor. Asserts
    # the copy has the same algorithm and state values. Then the same copy-and-compare is done
    # inside a traced function that creates the copy only once (module level holder).
    """Tests copying a generator."""
    g = random.Generator.from_seed(0)
    g_copy = random.Generator(g)
    self.assertAllEqual(g.algorithm, g_copy.algorithm)
    self.assertAllEqual(g.state.read_value(), g_copy.state.read_value())
    # Tests tf.function
    global g_seeded
    g_seeded = None
    # Do the same in tf.function
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function under test. Takes no arguments, returns nothing. On the first trace
      # copies the enclosing generator into the module level holder; asserts the copy has the same
      # algorithm and state values as the original.
      global g_seeded
      # defun'ed function should only create variables once
      if g_seeded is None:
        g_seeded = random.Generator(g)
      self.assertAllEqual(g.algorithm, g_seeded.algorithm)
      self.assertAllEqual(g.state.read_value(), g_seeded.state.read_value())
    f()

  @test_util.run_v1_only(
      ("This test is specifically for checking TF1 compatibility. "
       "It cannot run under TF2."))
  def testTF1(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode only): checks TF1 compatibility. A seeded generator and a
    # non-deterministic one are created, their state initializers are run in a session, and normal
    # samples are fetched twice. The seeded samples must match recorded constants within a small
    # tolerance and the unseeded sample must have the right shape.
    seed = 1234
    shape = [2, 3]
    expected_normal1 = constant_op.constant(
        [[0.9356609, 1.0854305, -0.93788373],
         [-0.50615472, 1.31697023, 0.71375787]], dtype=dtypes.float32)
    expected_normal2 = constant_op.constant(
        [[-0.3964749, 0.8369565, -0.30946946],
         [1.1206646, 1.00852597, -0.10185789]], dtype=dtypes.float32)
    with self.cached_session() as sess:
      gen1 = random.Generator.from_seed(seed)
      gen2 = random.Generator.from_non_deterministic_state()
      sess.run((gen1.state.initializer, gen2.state.initializer))
      r1 = gen1.normal(shape, dtype=dtypes.float32)
      r2 = gen2.normal(shape, dtype=dtypes.float32)
      def f():
        # [FUNCTION DOC]
        # HELPER: runs the two prepared sample tensors in the session and returns both results as a
        # pair. Takes no arguments.
        return sess.run((r1, r2))
      def check_results(expected_normal, v1, v2):
        # [FUNCTION DOC]
        # HELPER: asserts the first value is close to the expected samples (small relative and
        # absolute tolerance) and the second has the enclosing shape. Returns nothing.
        self.assertAllClose(expected_normal, v1, rtol=1e-5, atol=1e-5)
        self.assertAllEqual(shape, v2.shape)
      check_results(expected_normal1, *f())
      check_results(expected_normal2, *f())

  @test_util.run_v2_only
  @test_util.also_run_as_tf_function
  def testEagerAndDefun(self):
    # [FUNCTION DOC]
    # Smoke test (v2 only, also run wrapped as a tf.function): draws a normal sample of shape 3 from
    # the global generator and asserts nothing explicit; passes if no error is raised.
    """A simple test to make sure the op works in eager and defunned mode."""
    random.get_global_generator().normal((3,))

  @test_util.run_v2_only
  def testOpSeedSelectionAfterSetSeed(self):
    # [FUNCTION DOC]
    # Scenario (v2 only), regression for a GitHub issue: resetting the global generator from seed 1
    # and sampling gives the same values each time, both eagerly and inside a traced function,
    # and the eager and traced samples are equal to one another. Mutates the global generator.
    """Tests that op-seed selection is reset after reseting global generator.

    Fixing GitHub issue 9171:
    https://github.com/tensorflow/tensorflow/issues/9171
    """
    shape = (3,)
    random.get_global_generator().reset_from_seed(1)
    a = random.get_global_generator().normal(shape)
    random.get_global_generator().reset_from_seed(1)
    b = random.get_global_generator().normal(shape)
    self.assertAllEqual(a, b)

    # Now do the above again using accelerated ('defun'ed) computation
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function returning a normal sample of the enclosing shape from the current
      # global generator. Takes no arguments.
      return random.get_global_generator().normal(shape)

    random.get_global_generator().reset_from_seed(1)
    c = f()
    random.get_global_generator().reset_from_seed(1)
    d = f()
    self.assertAllEqual(c, d)
    self.assertAllEqual(a, c)

  @test_util.run_v2_only
  def testOpSeedSelectionNotSensitive(self):
    # [FUNCTION DOC]
    # Scenario (v2 only), regression for an internal bug: the sample from the global generator after
    # a reset must not depend on whether an unrelated print op was added to the graph. Compares
    # all combinations of with and without print, both eagerly and through a traced function that
    # is traced separately for each flag. Mutates the global generator.
    """Test that op-seed selection is not sensitive to trivial changes.

    Test that op-seed selection is not sensitive to trivial computation
    (i.e. graph) changes.

    Fixing b/32087099
    """
    def f(include_print):
      # [FUNCTION DOC]
      # HELPER: code under test in the op-seed insensitivity test.
      # Input: a boolean saying whether to wrap a constant shape tensor in a print op. Returns a
      # normal sample drawn from the global generator using that shape (the print op does not
      # change the values).
      # KNOWN ISSUE: relies on the deprecated print op from the logging ops module.
      shape = constant_op.constant([5])
      if include_print:
        shape = logging_ops.Print(shape, [shape])
      return random.get_global_generator().normal(shape)

    def compare(fst_includes_print, snd_includes_print):
      # [FUNCTION DOC]
      # HELPER: comparison step of the op-seed insensitivity test.
      # Inputs: two booleans choosing whether the first and second computation include a print op.
      # Resets the global generator to seed 50 before each computation, asserts the two eager
      # results are equal, then repeats with a traced version of the function and asserts equality
      # again. Returns nothing.
      random.get_global_generator().reset_from_seed(50)
      fst = f(fst_includes_print)
      random.get_global_generator().reset_from_seed(50)
      snd = f(snd_includes_print)
      self.assertAllEqual(fst, snd)
      # Now do the above again using accelerated (defunned) 'f'.
      # Running 'f' with two different Boolean arguments should cause
      # two different graphs to be generated, hence demonstrating the
      # insensitivity to graph changes.
      f_acc = def_function.function(f)
      random.get_global_generator().reset_from_seed(50)
      fst = f_acc(fst_includes_print)
      random.get_global_generator().reset_from_seed(50)
      snd = f_acc(snd_includes_print)
      self.assertAllEqual(fst, snd)

    compare(False, False)
    compare(True, True)
    compare(True, False)

  @parameterized.parameters(list(random_ops_util.Algorithm))
  @test_util.run_v2_only
  def testKey(self, alg):
    # [FUNCTION DOC]
    # Scenario (parameterized over algorithms, v2 only): a generator is built with a state whose key
    # slot holds 1234 (preceded by zeros for the counter). Asserts the key property returns 1234,
    # both eagerly and from inside a traced function.
    key = 1234
    gen = random.Generator(
        state=[0] * random._get_counter_size(alg.value) + [key], alg=alg)
    got = gen.key
    self.assertAllEqual(key, got)
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function returning the key property of the enclosing generator. Takes no
      # arguments.
      return gen.key
    got = f()
    self.assertAllEqual(key, got)

  @test_util.run_v2_only
  def testSkip(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): Philox generator with counter 5678 and key 1234; skip(432) is called.
    # Asserts the first state word becomes the old counter plus 432 times 256, i.e. each skipped
    # step advances the counter by 256.
    key = 1234
    counter = 5678
    gen = random.Generator(state=[counter, 0, key], alg=random.RNG_ALG_PHILOX)
    delta = 432
    gen.skip(delta)
    new_counter = gen.state[0]
    self.assertAllEqual(counter + delta * 256, new_counter)

  def _sameAsOldRandomOps(self, device, floats):
    # [FUNCTION DOC]
    # HELPER: shared body of the CPU and GPU same-as-old-ops tests.
    # Inputs: a device name string and a list of float dtypes to test. For each float dtype the new
    # generator based normal, truncated normal and uniform ops are compared against the legacy
    # random ops with the equivalent seeds, and for each integer dtype the uniform integer ops are
    # compared. Each comparison runs five times and asserts equal outputs. Returns nothing.
    def compare(dtype, old, new):
      # [FUNCTION DOC]
      # HELPER: inner comparison of _sameAsOldRandomOps.
      # Inputs: a dtype, a function producing output with the old op and one producing it with the
      # new generator. Creates a Philox generator on the device whose state encodes the same
      # two seeds used for the old op, and five times asserts the old (traced) and new outputs are
      # equal. Returns nothing.
      seed1, seed2 = 79, 25
      # note how the two seeds for the old op correspond to the seed for the new
      # op
      with ops.device(device):
        gen = random.Generator(state=[0, seed2, seed1],
                               alg=random.RNG_ALG_PHILOX)

      # create a graph for the old op in order to call it many times
      @def_function.function
      def run_old():
        # [FUNCTION DOC]
        # HELPER: traced wrapper so the old op can be called repeatedly. Takes no arguments;
        # returns the old op output computed on the chosen device with the fixed seeds.
        with ops.device(device):
          return old(dtype, seed1, seed2)

      def run_new():
        # [FUNCTION DOC]
        # HELPER: calls the new-generator function on the chosen device with the shared generator
        # and returns its output. Takes no arguments.
        with ops.device(device):
          return new(dtype, gen)

      for _ in range(5):
        self.assertAllEqual(run_old(), run_new())

    shape = constant_op.constant([4, 7])
    minval = 128
    maxval = 256

    # passing `dtype` around to compress go/gpylint-faq#cell-var-from-loop and
    # go/gpylint-faq#undefined-loop-variable
    def old_normal(dtype, seed1, seed2):
      # [FUNCTION DOC]
      # HELPER: returns the legacy standard normal op output of the shared shape for the dtype and
      # the two seeds.
      return gen_random_ops.random_standard_normal(
          shape, dtype=dtype, seed=seed1, seed2=seed2)
    def new_normal(dtype, gen):
      # [FUNCTION DOC]
      # HELPER: returns a standard normal sample of the shared shape from the generator using its
      # private standard-normal method.
      return gen._standard_normal(shape, dtype=dtype)
    def old_truncated_normal(dtype, seed1, seed2):
      # [FUNCTION DOC]
      # HELPER: returns the legacy truncated normal op output of the shared shape for the dtype and
      # the two seeds.
      return gen_random_ops.truncated_normal(
          shape, dtype=dtype, seed=seed1, seed2=seed2)
    def new_truncated_normal(dtype, gen):
      # [FUNCTION DOC]
      # HELPER: returns a truncated normal sample of the shared shape from the generator using its
      # private truncated-normal method.
      return gen._truncated_normal(shape, dtype=dtype)
    def old_uniform_int(dtype, seed1, seed2):
      # [FUNCTION DOC]
      # HELPER: builds min (128) and max (256) constants of the dtype and returns the legacy uniform
      # integer op output of the shared shape with the two seeds.
      minval2 = constant_op.constant(minval, dtype=dtype)
      maxval2 = constant_op.constant(maxval, dtype=dtype)
      return gen_random_ops.random_uniform_int(
          shape, minval=minval2, maxval=maxval2, seed=seed1, seed2=seed2)
    def new_uniform_int(dtype, gen):
      # [FUNCTION DOC]
      # HELPER: returns an integer uniform sample between 128 and 256 of the shared shape from the
      # generator public uniform method.
      return gen.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    def old_uniform(dtype, seed1, seed2):
      # [FUNCTION DOC]
      # HELPER: returns the legacy random uniform op output of the shared shape for the dtype and
      # the two seeds.
      return gen_random_ops.random_uniform(
          shape, dtype=dtype, seed=seed1, seed2=seed2)
    def new_uniform(dtype, gen):
      # [FUNCTION DOC]
      # HELPER: returns a uniform sample of the shared shape from the generator using its private
      # uniform method.
      return gen._uniform(shape, dtype=dtype)

    for dtype in floats:
      compare(dtype, old_normal, new_normal)
      compare(dtype, old_truncated_normal, new_truncated_normal)
      compare(dtype, old_uniform, new_uniform)
    for dtype in INTS:
      compare(dtype, old_uniform_int, new_uniform_int)

  @test_util.run_v2_only
  def testSameAsOldRandomOpsCPU(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): runs the shared old-versus-new comparison on the first CPU device with
    # the full list of float dtypes (including bfloat16) and integer dtypes.
    """Tests that the generated numbers are the same as the old random_ops.py.

    The CPU version.
    """
    self._sameAsOldRandomOps("/device:CPU:0", FLOATS)

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testSameAsOldRandomOpsGPU(self):
    # [FUNCTION DOC]
    # Scenario (v2 and CUDA only): same comparison on the GPU device. Float dtypes are half, single
    # and double precision, plus bfloat16 only when the GPU has compute capability 8.0 or higher.
    """Tests that the generated numbers are the same as the old random_ops.py.

    The GPU version.
    """
    floats = [dtypes.float16, dtypes.float32, dtypes.float64]
    if test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      floats += [dtypes.bfloat16]
    self._sameAsOldRandomOps(test_util.gpu_device_name(), floats)

  @parameterized.parameters(INTS + [dtypes.uint32, dtypes.uint64])
  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testGPUEqualsCPU(self, dtype):
    # [FUNCTION DOC]
    # Scenario (parameterized over int32, int64, uint32, uint64; v2 and CUDA only): the same seed
    # draws full-range integers of shape 315 by 49 on CPU and on GPU; the outputs must be equal.
    """Tests that GPU and CPU generate the same integer outputs."""
    seed = 1234
    shape = [315, 49]
    with ops.device("/device:CPU:0"):
      cpu = random.Generator.from_seed(seed).uniform_full_int(
          shape=shape, dtype=dtype)
    with ops.device(test_util.gpu_device_name()):
      gpu = random.Generator.from_seed(seed).uniform_full_int(
          shape=shape, dtype=dtype)
    self.assertAllEqual(cpu, gpu)

  @parameterized.parameters(FLOATS + INTS)
  @test_util.run_v2_only
  def testUniformIsInRange(self, dtype):
    # [FUNCTION DOC]
    # Scenario (parameterized over float and int dtypes; v2 only): skips bfloat16 when the GPU lacks
    # compute capability 8.0. Draws 1000 uniform values with bounds 2 (inclusive) and 33
    # (exclusive) and asserts all values are at least the minimum and below the maximum.
    if dtype == dtypes.bfloat16 and not test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      self.skipTest("Bfloat16 requires compute capability 8.0")
    minval = 2
    maxval = 33
    size = 1000
    gen = random.Generator.from_seed(1234)
    x = gen.uniform(
        shape=[size], dtype=dtype, minval=minval, maxval=maxval).numpy()
    self.assertTrue(np.all(x >= minval))
    self.assertTrue(np.all(x < maxval))

  @parameterized.parameters(FLOATS)
  @test_util.run_v2_only
  def testNormalIsFinite(self, dtype):
    # [FUNCTION DOC]
    # Scenario (parameterized over float dtypes; v2 only): skips bfloat16 without suitable GPU.
    # Draws 10000 normal values and asserts every one is finite.
    if dtype == dtypes.bfloat16 and not test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      self.skipTest("Bfloat16 requires compute capability 8.0")
    gen = random.Generator.from_seed(1234)
    x = gen.normal(shape=[10000], dtype=dtype).numpy()
    self.assertTrue(np.all(np.isfinite(x)))

  @parameterized.parameters(FLOATS + INTS)
  @test_util.run_v2_only
  def testDistributionOfUniform(self, dtype):
    # [FUNCTION DOC]
    # Scenario (parameterized over float and int dtypes; v2 only): draws 1000 uniform samples with a
    # fixed seed (maximum 1 for floats, 100 for ints, normalized to the unit range), computes a
    # chi-squared statistic over 10 bins and asserts it is below 16.92 (the 5 percent critical
    # value for 9 degrees of freedom). Skips bfloat16 without suitable GPU.
    """Use Pearson's Chi-squared test to test for uniformity."""
    if dtype == dtypes.bfloat16 and not test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      self.skipTest("Bfloat16 requires compute capability 8.0")
    n = 1000
    seed = 123
    gen = random.Generator.from_seed(seed)
    maxval = 1
    if dtype.is_integer:
      maxval = 100
    x = gen.uniform(shape=[n], maxval=maxval, dtype=dtype).numpy()
    if maxval > 1:
      # Normalize y to range [0, 1).
      x = x.astype(float) / maxval
    # Tests that the values are distributed amongst 10 bins with equal
    # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
    # p=0.05. This test is probabilistic and would be flaky if the random
    # seed were not fixed.
    val = random_test_util.chi_squared(x, 10)
    self.assertLess(val, 16.92)

  @parameterized.parameters(FLOATS)
  @test_util.run_v2_only
  def testDistributionOfNormal(self, dtype):
    # [FUNCTION DOC]
    # Scenario (parameterized over float dtypes; v2 only): draws 1000 normal samples with a fixed
    # seed and asserts the Anderson-Darling statistic is below 2.492 (5 percent critical value).
    # Skips bfloat16 without suitable GPU.
    """Use Anderson-Darling test to test distribution appears normal."""
    if dtype == dtypes.bfloat16 and not test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      self.skipTest("Bfloat16 requires compute capability 8.0")
    n = 1000
    gen = random.Generator.from_seed(1234)
    x = gen.normal(shape=[n], dtype=dtype).numpy()
    # The constant 2.492 is the 5% critical value for the Anderson-Darling
    # test where the mean and variance are known. This test is probabilistic
    # so to avoid flakiness the seed is fixed.
    self.assertLess(
        random_test_util.anderson_darling(x.astype(float)), 2.492)

  @test_util.run_v2_only
  def testErrors(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): drives the raw stateful random ops with invalid inputs and asserts the
    # error type and message: a non-scalar algorithm argument or skip delta (invalid-argument
    # error about shape), a float algorithm (type error about int64 eager tensor), an unsupported
    # algorithm id for normal sampling and for read-and-skip, a state variable of the wrong dtype,
    # of the wrong rank, or too small for Philox, and (via traced functions) uniform called with a
    # non-scalar minval or maxval (value error).
    """Tests that proper errors are raised.
    """
    shape = [2, 3]
    gen = random.Generator.from_seed(1234)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        r"must have shape \[\], not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, [0, 0], shape)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        r"must have shape \[\], not"):
      gen_stateful_random_ops.rng_skip(
          gen.state.handle, gen.algorithm, [0, 0])
    with self.assertRaisesWithPredicateMatch(
        TypeError, "EagerTensor of dtype int64"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 1.1, shape)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "Unsupported algorithm id"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 123, shape)
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "Unsupported algorithm id"):
      gen_stateful_random_ops.rng_read_and_skip(
          gen.state.handle, alg=123, delta=10)
    var = variables.Variable([0, 0], dtype=dtypes.int32)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "dtype of RNG state variable must be int64, not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)
    var = variables.Variable([[0]], dtype=dtypes.int64)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "RNG state must have one and only one dimension, not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)
    var = variables.Variable([0], dtype=dtypes.int64)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "For the Philox algorithm, the size of state must be at least"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        "minval must be a scalar; got a tensor of shape "):
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: traced function that calls uniform with a non-scalar tensor as the lower bound for
        # int32; expected to fail with a value error during tracing. Takes no arguments.
        gen.uniform(shape=shape, minval=array_ops.zeros(shape, "int32"),
                    maxval=100, dtype="int32")
      f()
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        "maxval must be a scalar; got a tensor of shape "):
      @def_function.function
      def f2():
        # [FUNCTION DOC]
        # HELPER: traced function that calls uniform with a non-scalar tensor as the upper bound for
        # int32; expected to fail with a value error during tracing. Takes no arguments.
        gen.uniform(
            shape=shape, minval=0, maxval=array_ops.ones(shape, "int32") * 100,
            dtype="int32")
      f2()

  @test_util.run_v2_only
  def testGetGlobalGeneratorWithXla(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test calls skipTest unconditionally on its first statements, so everything
    # after it (the XLA device check and the compiled-generator code) is dead and never runs.
    # Intended scenario: with the global generator cleared, a jit-compiled function reads the
    # global generator state and draws full-range integers on an XLA CPU device; the seed must be
    # finite and reproducible after resetting the generator to the saved state.
    """Demonstrates using the global generator with XLA."""
    # This test was passing before because soft placement silently picked the
    # CPU kernel.
    # TODO(wangpeng): Remove this skip
    self.skipTest("NonDeterministicInts lacks XLA kernel.")

    if not config.list_physical_devices("XLA_CPU"):
      self.skipTest("No XLA_CPU device available.")

    random.set_global_generator(None)

    @def_function.function(jit_compile=True)
    def make_seed():
      # [FUNCTION DOC]
      # HELPER: jit-compiled function used by the XLA global generator test (never reached because
      # the test skips). Takes no arguments; returns a pair of a two-element int32 sample drawn from
      # the global generator and a copy of the generator state read before sampling.
      generator = random.get_global_generator()
      state = array_ops.identity(generator.state, name="state")
      return generator.uniform_full_int((2,), dtypes.int32, name="seed"), state

    with ops.device("/device:XLA_CPU:0"):
      seed, state = make_seed()
      self.assertTrue(np.all(np.isfinite(seed.numpy())))
      random.get_global_generator().reset(state)
      self.assertAllEqual(make_seed()[0], seed)

  @test_util.run_v2_only
  def testSetGlobalGeneratorBadWithDefun(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): a traced function samples from the global generator. After the traced
    # function captured the first global generator, a new global generator with the same seed is
    # set. Asserts the next call returns different samples, i.e. replacing the global generator
    # does not affect an already compiled function (documented as a bad pattern).
    """Demonstrates set_global_generator does not affect compiled tf.function."""
    shape = (3,)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function returning a normal sample of the enclosing shape from the global
      # generator that is current at trace time. Takes no arguments.
      return random.get_global_generator().normal(shape)

    random.set_global_generator(random.Generator.from_seed(50))
    samples = f()
    # Resetting global generator has no effect to the compiled tf.function.
    random.set_global_generator(random.Generator.from_seed(50))
    # New samples are returned.
    self.assertNotAllEqual(samples, f())

  @test_util.run_v2_only
  def testFunctionArg(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): a generator is passed as an argument to a traced function that draws a
    # normal sample. A second generator with the same seed draws eagerly. Asserts equal samples
    # and equal final states.
    """Tests that RNG can be used as tf.function's argument.
    """
    shape = [2, 3]
    @def_function.function
    def f(gen):
      # [FUNCTION DOC]
      # HELPER: traced function under test. Input: a generator object passed as an argument.
      # Returns a normal sample of the enclosing shape from it.
      return gen.normal(shape)
    g1 = random.Generator.from_seed(1)
    g2 = random.Generator.from_seed(1)
    res1 = f(g1)
    res2 = g2.normal(shape)
    self.assertAllEqual(res1, res2)
    self.assertAllEqual(g1.state.read_value(), g2.state.read_value())

  @test_util.run_v2_only
  def testUniformFullInt(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): uniform with minval set to nothing for int32 must produce the same values
    # as uniform_full_int from an identically seeded generator.
    """Tests full-range int uniform.
    """
    shape = [3, 4]
    dtype = dtypes.int32
    g = random.Generator.from_seed(1)
    r1 = g.uniform(shape=shape, dtype=dtype, minval=None)
    g = random.Generator.from_seed(1)
    r2 = g.uniform_full_int(shape=shape, dtype=dtype)
    self.assertAllEqual(r1, r2)

  @test_util.run_v2_only
  def testRestore(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): a generator is tracked by a checkpoint. The checkpoint is written to a
    # file under the test temp directory, a value is drawn, the checkpoint is restored and a value
    # drawn again; the two values must be equal. Repeated twice so writing happens in different
    # generator states. Side effect: writes a checkpoint file.
    """Tests save and restore.
    """
    fname = os.path.join(self.get_temp_dir(), "checkpoint")
    g = random.Generator.from_seed(1)
    cp = tracking_util.Checkpoint(g=g)
    def write_restore_compare():
      # [FUNCTION DOC]
      # HELPER: one round of checkpoint write, draw, restore, draw for testRestore. Asserts the two
      # draws are equal. Takes no arguments; returns nothing; writes the checkpoint file.
      cp.write(fname)
      r1 = g.uniform([], dtype=dtypes.uint32, minval=None)
      cp.restore(fname)
      r2 = g.uniform([], dtype=dtypes.uint32, minval=None)
      self.assertAllEqual(r1, r2)
    # Run multiple times so that cp.write is called in various RNG states
    for _ in range(2):
      write_restore_compare()

  @test_util.run_v2_only
  def testDeterministicOpsErrors(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): enables op determinism and clears the global generator. Getting the global
    # generator must raise a runtime error; after setting a seeded global generator getting it
    # works; creating a generator from non-deterministic state raises a runtime error. A finally
    # step always turns determinism back off.
    try:
      config.enable_op_determinism()
      random.set_global_generator(None)
      with self.assertRaisesWithPredicateMatch(
          RuntimeError,
          '"get_global_generator" cannot be called if determinism is enabled'):
        random.get_global_generator()
      random.set_global_generator(random.Generator.from_seed(50))
      random.get_global_generator()
      with self.assertRaisesWithPredicateMatch(
          RuntimeError,
          '"from_non_deterministic_state" cannot be called when determinism '
          "is enabled."):
        random.Generator.from_non_deterministic_state()
    finally:
      config.disable_op_determinism()


if __name__ == "__main__":
  config.set_soft_device_placement(False)
  test.main()
