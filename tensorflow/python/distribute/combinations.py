# [MODULE DOC]
# Test-combination support for tf.distribute strategies. It specializes the generic
# test_combinations framework with parameter modifiers and combinations that turn
# named strategy descriptors into live strategy objects, fill in multi-worker cluster
# parameters, and skip combinations that need GPUs/TPUs/XLA that are not available.
# Exposes generate, combine, times, NamedObject, NamedDistribution, concat, the
# tf_function / no_tf_function named objects, and environment helpers (env,
# in_main_process). A multi-worker test is run by spawning worker processes that rerun
# the same test and report results back to the main process.
# Import-time side effects: inspects the program name (command line) to detect GPU/TPU
# test targets, creates a module-level test environment object, and registers a
# callback to run after application init that records the number of physical GPUs.
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
"""This module customizes `test_combinations` for `tf.distribute.Strategy`.

Additionally it provides `generate()`, `combine()` and `times()` with
`tf.distribute.Strategy` customizations as a default.
"""

import collections
import copy
import re
import sys
import types
import unittest

from absl import app
import six


from tensorflow.python.client import session
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


# TODO(rchao): Rename `distribution` parameter to `strategy` or
# `distribute_strategy` in all tests.
class DistributionParameter(combinations_lib.ParameterModifier):
  # [CLASS DOC]
  # Parameter modifier that replaces every argument holding a named distribution
  # descriptor with the strategy object it creates. Stateless; used by the
  # distribution combination.
  """Transforms arguments of type `NamedDistribution`.

  Convert all arguments of type `NamedDistribution` to the value of their
  `strategy` property.
  """

  def modified_arguments(self, kwargs, requested_parameters):
    # [FUNCTION DOC]
    # Builds the mapping of replacement arguments for a test combination.
    # Inputs: kwargs is the mapping of argument names to values for the combination;
    # requested_parameters is the list of parameter names the test wants (unused).
    # Output: a mapping from each argument name whose value is a named distribution to
    # the strategy object produced by that distribution (each access creates a new
    # strategy). If the optional use_var_policy argument is present and truthy, it is
    # also written onto each strategy's extended object as its variable-policy flag
    # (a side effect on the strategy). Raises nothing itself.
    # Get the parameter that indicates if we need to set the `_use_policy` flag
    # on the strategy object. This is a temporary flag for testing the variable
    # policy rollout.
    use_var_policy = kwargs.get("use_var_policy", None)
    distribution_arguments = {}
    for k, v in kwargs.items():
      if isinstance(v, NamedDistribution):
        strategy = v.strategy
        if use_var_policy:
          strategy.extended._use_var_policy = use_var_policy
        distribution_arguments[k] = strategy
    return distribution_arguments


class ClusterParameters(combinations_lib.ParameterModifier):
  # [CLASS DOC]
  # Parameter modifier that supplies multi-worker cluster parameters (chief flag,
  # worker count, runner, GPU sharing, parameter-server count) to a test, taking them
  # from a named distribution when one is present or from explicit arguments / defaults
  # otherwise. Stateless; used by the cluster combination.
  """Adds cluster parameters if a `NamedDistribution` has it.

  It needs to be before DistributionParameter.
  """

  def modified_arguments(self, kwargs, requested_parameters):
    # [FUNCTION DOC]
    # Computes cluster parameter values for a combination.
    # Inputs: kwargs is the mapping of argument names to values; requested_parameters is
    # the collection of parameter names the test function asks for.
    # Behavior: finds the named distribution among the arguments (the last one wins) and
    # raises a value error if a second one is found while the cluster has more than one
    # worker in total. If a distribution exists, its chief flag, worker count, runner,
    # GPU-sharing flag and parameter-server count are used; a value error is raised if
    # explicit chief or worker arguments disagree with it. Without a distribution, the
    # explicit arguments are used with defaults: no chief, one worker, no runner, share
    # GPUs, zero parameter servers.
    # Output: a mapping containing only those of the five cluster parameters that the
    # test requested. Creates no strategy; reading the runner may invoke a pool-runner
    # factory.
    strategy = None
    for _, v in kwargs.items():
      if isinstance(v, NamedDistribution):
        if strategy is not None and _num_total_workers(v.has_chief,
                                                       v.num_workers) > 1:
          raise ValueError("Only support one NamedDistribution for multi worker"
                           "tests.")
        strategy = v

    if strategy:
      has_chief = strategy.has_chief
      num_workers = strategy.num_workers
      runner = strategy.runner
      share_gpu = strategy.share_gpu
      num_ps = strategy.num_ps
      if "has_chief" in kwargs and kwargs["has_chief"] != has_chief:
        raise ValueError(
            "both has_chief and strategy specified but are not compatible")
      if "num_workers" in kwargs and kwargs["num_workers"] != num_workers:
        raise ValueError(
            "both num_workers and strategy specified but are not compatible")
    else:
      has_chief = kwargs.get("has_chief", False)
      num_workers = kwargs.get("num_workers", 1)
      runner = kwargs.get("runner", None)
      share_gpu = kwargs.get("share_gpu", True)
      num_ps = kwargs.get("num_ps", 0)

    # Always set cluster parameters if they're requested. So that generate()
    # works when there's no strategy in the combinations.
    update = {}
    if "has_chief" in requested_parameters:
      update["has_chief"] = has_chief
    if "num_workers" in requested_parameters:
      update["num_workers"] = num_workers
    if "runner" in requested_parameters:
      update["runner"] = runner
    if "share_gpu" in requested_parameters:
      update["share_gpu"] = share_gpu
    if "num_ps" in requested_parameters:
      update["num_ps"] = num_ps
    return update


class DistributionCombination(combinations_lib.TestCombination):
  # [CLASS DOC]
  # Test combination that wires distribution strategies into tests: it converts named
  # distributions into strategies, accepts an optional variable-policy argument, and
  # filters out combinations that cannot run under XLA. Stateless.
  """Sets up distribution strategy for tests."""

  def should_execute_combination(self, kwargs):
    # [FUNCTION DOC]
    # Decides whether a combination should run. Input: kwargs, mapping of argument names
    # to values. Output: a pair of (boolean, reason). Returns false with an explanatory
    # message when XLA testing is enabled and any named distribution is marked as not
    # supporting XLA; otherwise returns true with no reason.
    distributions = [
        v for v in kwargs.values() if isinstance(v, NamedDistribution)
    ]
    if test_util.is_xla_enabled() and any(d.no_xla for d in distributions):
      return (
          False,
          "n/a: skipping strategy combination with no_xla=True in XLA tests")
    return (True, None)

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Returns the list of parameter modifiers for this combination: the distribution
    # parameter modifier and an optional-parameter modifier for the variable-policy
    # argument (so tests may omit it).
    return [
        DistributionParameter(),
        combinations_lib.OptionalParameter("use_var_policy"),
    ]


class ClusterCombination(combinations_lib.TestCombination):
  # [CLASS DOC]
  # Test combination enabling multi-worker tests by contributing the cluster parameter
  # modifier. Stateless.
  """Sets up multi worker tests."""

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Returns a list holding a single cluster parameter modifier. Must be applied before
    # the distribution modifier takes effect on the named distributions.
    return [ClusterParameters()]


class GPUCombination(combinations_lib.TestCombination):
  # [CLASS DOC]
  # Test combination that lets tests declare GPU requirements (logical or physical GPU
  # counts, directly or via a named distribution) and skips combinations whose
  # requirements the machine cannot meet. Class attribute GPU_TEST is true when the
  # running program name looks like a GPU test target; it is computed once at class
  # creation from the command line.
  """Enable tests to request GPU hardware and skip non-GPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_gpus` argument is supported.  GPU hardware is
  required, if its value is `True` or > 0.

  Attributes:
    GPU_TEST: The environment is considered to have GPU hardware available if
              the name of the program contains "test_gpu" or "test_xla_gpu".
  """
  GPU_TEST = False
  if sys.argv:
    GPU_TEST = re.search(r"(test_2?gpu|test_xla_2?gpu)$", sys.argv[0])

  def should_execute_combination(self, kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the physical-GPU shortage message, the first count printed is the
    # list of physical GPU devices rather than its length, and the second is the
    # directly requested physical count rather than the computed requirement.
    # Decides whether a combination can run given GPU needs. Input: kwargs mapping of
    # argument names to values, which may include the required GPU counts and named
    # distributions. Output: pair of (boolean, reason).
    # Raises a value error if a distribution is combined with a logical GPU requirement,
    # or if both logical and physical requirements are given directly. The requirement is
    # the maximum over direct arguments and distributions. Returns false when no GPUs are
    # required but the program is a GPU test target, when fewer logical GPUs are
    # available than required, or when fewer physical GPUs are available than required;
    # otherwise true with no reason.
    distributions = [
        v for v in kwargs.values() if isinstance(v, NamedDistribution)
    ]
    required_gpus = kwargs.get("required_gpus", 0)
    required_physical_gpus = kwargs.get("required_physical_gpus", 0)

    if distributions and required_gpus:
      raise ValueError("Do not use `required_gpus` and arguments of type "
                       "NamedDistribution together.")

    number_of_required_gpus = max(
        [required_gpus] + [required_physical_gpus] +
        [d.required_physical_gpus or 0 for d in distributions] +
        [d.required_gpus or 0 for d in distributions])
    number_of_required_physical_gpus = max(
        [required_physical_gpus] +
        [d.required_physical_gpus or 0 for d in distributions])

    if (required_physical_gpus and required_gpus):
      raise ValueError("Only one of `required_physical_gpus`(number of physical"
                       " GPUs required) and `required_gpus`(total number of "
                       "GPUs required) should be set. ")
    if not number_of_required_gpus and GPUCombination.GPU_TEST:
      return (False, "Test that doesn't require GPUs.")
    elif (number_of_required_gpus > 0
          and context.num_gpus() < number_of_required_gpus):
      return (False, ("Only {} of {} required GPUs are available.".format(
          context.num_gpus(), number_of_required_gpus)))
    elif number_of_required_physical_gpus > len(
        config.list_physical_devices("GPU")):
      return (False,
              ("Only {} of {} required physical GPUs are available.".format(
                  config.list_physical_devices("GPU"), required_physical_gpus)))
    else:
      return (True, None)

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Returns optional-parameter modifiers for the logical and physical GPU requirement
    # arguments, so tests may omit them.
    return [combinations_lib.OptionalParameter("required_gpus"),
            combinations_lib.OptionalParameter("required_physical_gpus")]


class TPUCombination(combinations_lib.TestCombination):
  # [CLASS DOC]
  # Test combination that lets tests declare TPU requirements (directly or via a named
  # distribution) and filters combinations so TPU tests run only in TPU targets and
  # cloud versus local TPU usage matches the command-line setting. Class attribute
  # TPU_TEST is true when the program name contains the TPU test marker; computed once
  # at class creation.
  """Allow to request TPU hardware and skip non-TPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_tpus` parameter is supported.  TPU hardware is
  required, if its argument is `True` or > 0.

  Optionally, the `use_cloud_tpu` parameter is supported. If TPU hardware is
  required by `required_tpus`, it specifically must be a Cloud TPU (specified
  with `--tpu`) if `use_cloud_tpu` is `True`.

  Attributes:
    TPU_TEST: The environment is considered to have TPU hardware available if
              the name of the program contains "test_tpu".
  """

  TPU_TEST = False
  if sys.argv:
    TPU_TEST = "test_tpu" in sys.argv[0]

  def should_execute_combination(self, kwargs):
    # [FUNCTION DOC]
    # Decides whether a combination can run given TPU needs. Input: kwargs mapping of
    # argument names to values (required TPU flag or count under either of two names,
    # cloud-TPU flag, named distributions). Output: pair of (boolean, reason).
    # Raises a value error if both the old and new TPU argument names are given, or if a
    # distribution is combined with a direct TPU requirement. Requirement is binary
    # (any nonzero value), taking the maximum over direct argument and distributions;
    # cloud usage is true if requested directly or by any distribution. Reads the
    # optional tpu command-line flag. Returns false when: no TPU is required but this is
    # a TPU target; a TPU is required but this is not a TPU target; cloud TPU is wanted
    # but none specified; local TPU is wanted but a cloud TPU is specified. Otherwise
    # true with no reason.
    distributions = [
        v for v in kwargs.values() if isinstance(v, NamedDistribution)
    ]
    # TODO(isaprykin): Migrate all tests away from using 'required_tpu' in favor
    # of 'required_tpus'.
    if "required_tpus" in kwargs and "required_tpu" in kwargs:
      raise ValueError("Do not use `required_tpu`.  Both `required_tpus` and "
                       "`required_tpu` were specified.")
    required_tpus = kwargs.get("required_tpus", None) or kwargs.get(
        "required_tpu", None)

    if distributions and required_tpus:
      raise ValueError("Do not use `required_tpus` and arguments of type "
                       "NamedDistribution together.")

    # TODO(isaprykin): Add support for a particular number of TPUs.  Right now
    # it's binary.
    number_of_required_tpus = max([required_tpus or 0] +
                                  [d.required_tpu or 0 for d in distributions])
    use_cloud_tpu = any([kwargs.get("use_cloud_tpu")] +
                        [d.use_cloud_tpu for d in distributions])
    tpu = hasattr(flags.FLAGS, "tpu") and flags.FLAGS.tpu or ""

    if not number_of_required_tpus and TPUCombination.TPU_TEST:
      return (False, "Test that doesn't require TPUs.")
    if number_of_required_tpus and not TPUCombination.TPU_TEST:
      return (False, "Test requires a TPU, but it's not available.")
    if use_cloud_tpu and not tpu:
      return (False, "Test requires a Cloud TPU, but none specified.")
    if not use_cloud_tpu and tpu:
      return (False, "Test requires local TPU, but Cloud TPU specified.")
    return (True, None)

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Returns optional-parameter modifiers for the TPU count (new name), the legacy TPU
    # flag name, and the cloud-TPU flag, so tests may omit them.
    return [
        combinations_lib.OptionalParameter("required_tpus"),
        combinations_lib.OptionalParameter("required_tpu"),
        combinations_lib.OptionalParameter("use_cloud_tpu"),
    ]


class NamedDistribution(object):
  # [CLASS DOC]
  # Descriptor for a distribution strategy used in test combinations. Holds a display
  # name (used as the test title), a factory that creates the strategy on demand, and the
  # hardware/cluster requirements (GPUs, physical GPUs, TPU, cloud TPU, chief, worker
  # count, parameter servers, GPU sharing, optional pool-runner factory, XLA exclusion).
  # Strategies are created lazily, anew each time the strategy property is read.
  """Wraps a `tf.distribute.Strategy` and adds a name for test titles."""

  def __init__(self,
               name,
               distribution_fn,
               required_gpus=None,
               required_physical_gpus=0,
               required_tpu=False,
               use_cloud_tpu=False,
               has_chief=False,
               num_workers=1,
               num_ps=0,
               share_gpu=True,
               pool_runner_fn=None,
               no_xla=False):
    # [FUNCTION DOC]
    # Constructor. Stores the display name, the strategy factory and all requirement
    # settings as given. name is a string used in test titles; distribution_fn is a
    # callable returning a strategy; required_gpus (optional, number or nothing) and
    # required_physical_gpus (default zero) are GPU needs; required_tpu and use_cloud_tpu
    # are booleans (default false); has_chief (default false), num_workers (default one),
    # num_ps (default zero) describe the cluster; share_gpu (default true) says whether
    # workers share GPUs; pool_runner_fn is an optional callable returning a pool runner;
    # no_xla (default false) excludes the distribution from XLA test runs. Nothing is
    # validated or created here and nothing is returned.
    """Initialize NamedDistribution.

    Args:
      name: Name that will be a part of the name of the test case.
      distribution_fn: A callable that creates a `tf.distribute.Strategy`.
      required_gpus: The number of GPUs that the strategy requires. Only one of
      `required_gpus` and `required_physical_gpus` should be set.
      required_physical_gpus: Number of physical GPUs required. Only one of
      `required_gpus` and `required_physical_gpus` should be set.
      required_tpu: Whether the strategy requires TPU.
      use_cloud_tpu: Whether the strategy requires cloud TPU.
      has_chief: Whether the strategy requires a chief worker.
      num_workers: The number of workers that the strategy requires.
      num_ps: The number of parameter servers.
      share_gpu: Whether to share GPUs among workers.
      pool_runner_fn: An optional callable that returns a MultiProcessPoolRunner
        to run the test.
      no_xla: Whether to skip in XLA tests.
    """
    object.__init__(self)
    self._name = name
    self._distribution_fn = distribution_fn
    self.required_gpus = required_gpus
    self.required_physical_gpus = required_physical_gpus
    self.required_tpu = required_tpu
    self.use_cloud_tpu = use_cloud_tpu
    self.has_chief = has_chief
    self.num_workers = num_workers
    self.num_ps = num_ps
    self.share_gpu = share_gpu
    self._pool_runner_fn = pool_runner_fn
    self.no_xla = no_xla

  @property
  def runner(self):
    # [FUNCTION DOC]
    # Property. Returns a pool runner created by calling the pool-runner factory when one
    # was supplied (a new call each access), otherwise nothing.
    if self._pool_runner_fn is not None:
      return self._pool_runner_fn()
    return None

  @property
  def strategy(self):
    # [FUNCTION DOC]
    # Property. Calls the strategy factory and returns the newly created strategy; every
    # access builds a fresh strategy (and may create devices or servers as a side effect).
    return self._distribution_fn()

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the display name, so test titles show the distribution's name.
    return self._name


# This is to allow adding combinations that runs a function both as a
# tf.function and eagerly.
#
# @combinations.generate(
#   combinations.combine(
#     tf_function = [combinations.tf_function, combinations.no_tf_function]
#   )
# )
# def testXXX(tf_function):
#   @tf_function
#   def foo():
#     tf.add(1., 1.)
#
#   foo()
tf_function = combinations_lib.NamedObject("TfFunction", def_function.function)
no_tf_function = combinations_lib.NamedObject("NoTfFunction", lambda f: f)


def concat(*combined):
  # [FUNCTION DOC]
  # Concatenates any number of combination lists into one new list, preserving order.
  # Inputs: zero or more lists. Output: the combined list. Inputs are not modified.
  """Concats combinations."""
  result = []
  for one in combined:
    result += one
  return result


@tf_export("__internal__.distribute.combinations.generate", v1=[])
def generate(combinations, test_combinations=()):
  # [FUNCTION DOC]
  # Distributed-aware test generator. Inputs: combinations, the list of argument
  # combinations to run (as produced by combine/times); test_combinations, an optional
  # tuple of extra test combinations appended after the defaults (eager/graph mode,
  # TF version, cluster, distribution, GPU, TPU). Output: a decorator that can be applied
  # to a test method or a test class. For a class, every attribute whose name starts with
  # the unittest test prefix and is a plain function is wrapped for multi-worker execution
  # (the class is modified in place); for a method, it is wrapped likewise. The wrapped
  # result is then passed through the generic combination generator, which produces the
  # parameterized tests. Exported as part of the internal distribute combinations API.
  # pylint: disable=g-doc-args,g-doc-return-or-yield
  """Distributed adapter of `tf.__internal__.test.combinations.generate`.

  All tests with distributed strategy should use this one instead of
  `tf.__internal__.test.combinations.generate`. This function has support of
  strategy combinations, GPU/TPU and multi worker support.

  See `tf.__internal__.test.combinations.generate` for usage.
  """
  # pylint: enable=g-doc-args,g-doc-return-or-yield
  default_combinations = (
      framework_combinations.EagerGraphCombination(),
      framework_combinations.TFVersionCombination(),
      ClusterCombination(),
      DistributionCombination(),
      GPUCombination(),
      TPUCombination(),
  )
  # We apply our own decoration to handle multi worker tests before applying
  # framework.test_combinations.generate. The order is important since we need
  # framework.test_combinations.generate to apply all parameter modifiers first.
  combination_decorator = combinations_lib.generate(
      combinations, test_combinations=default_combinations + test_combinations)

  def decorator(test_method_or_class):
    # [FUNCTION DOC]
    # HELPER: inner decorator produced by generate.
    # Applies multi-worker wrapping and the combination decorator to a test class (all
    # test-prefixed plain functions are replaced in place on the class) or to a single
    # test method. Returns the decorated class or method.
    if isinstance(test_method_or_class, type):
      # If it's a test class.
      class_object = test_method_or_class
      # Decorate each test method with _multi_worker_test.
      for name, test_method in six.iteritems(class_object.__dict__.copy()):
        if (name.startswith(unittest.TestLoader.testMethodPrefix) and
            isinstance(test_method, types.FunctionType)):
          setattr(class_object, name, _multi_worker_test(test_method))
      return combination_decorator(class_object)
    else:
      return combination_decorator(_multi_worker_test(test_method_or_class))

  return decorator


combine = combinations_lib.combine
times = combinations_lib.times
NamedObject = combinations_lib.NamedObject


# Identifies whether we're in the main process or worker processes.
# `_multi_worker_test` decoration behaves differently in the main processes and
# the worker processes. See the documentation of _multi_worker_test for detail.
_running_in_worker = False


@tf_export("__internal__.distribute.combinations.in_main_process", v1=[])
def in_main_process():
  # [FUNCTION DOC]
  # Returns true when running in the main test process, false inside a spawned worker
  # process. Takes no inputs and has no side effects.
  """Whether it's in the main test process.

  This is normally used to prepare the test environment which should only happen
  in the main process.

  Returns:
    A boolean.
  """
  return not _running_in_worker


class TestEnvironment(object):
  # [CLASS DOC]
  # Container for test environment information shared from the main process to worker
  # processes: the data-service dispatcher address (initially nothing) and the total
  # number of physical GPUs (initially nothing). Attribute assignment is only allowed in
  # the main process. A single module-level instance is created and exposed via env.
  """Holds the test environment information.

  Tests should modify the attributes of the instance returned by `env()` in the
  main process if needed, and it will be passed to the worker processes each
  time a test case is run.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Sets the data-service dispatcher address and the total physical GPU
    # count to nothing. Raises a value error if invoked outside the main process (via the
    # guarded attribute setter).
    self.tf_data_service_dispatcher = None
    # Note that this includes GPUs that may not be visible to the current
    # worker.
    self.total_phsyical_gpus = None

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Guarded attribute assignment. Inputs: attribute name and value. Raises a value error
    # when not running in the main process (message tells to condition on the main-process
    # check); otherwise performs the normal assignment. Returns nothing.
    if not in_main_process():
      raise ValueError(
          "combinations.env() should only be modified in the main process. "
          "Condition your code on combinations.in_main_process().")
    super().__setattr__(name, value)


_env = TestEnvironment()


@tf_export("__internal__.distribute.combinations.env", v1=[])
def env():
  # [FUNCTION DOC]
  # Returns the module-level test environment object (the one passed to workers; inside a
  # worker it is the copy received from the main process). No inputs, no side effects.
  """Returns the object holds the test environment information.

  Tests should modify this in the main process if needed, and it will be passed
  to the worker processes each time a test case is run.

  Returns:
    a TestEnvironment object.
  """
  return _env


def _set_total_phsyical_gpus():
  # [FUNCTION DOC]
  # HELPER: init-time callback that fills the shared test environment.
  # In the main process only, records on the test environment the number of physical GPUs
  # reported by the eager context. Takes no inputs, returns nothing. Registered to run
  # after application initialization so lazily loaded CUDA is accounted for.
  if in_main_process():
    env().total_phsyical_gpus = len(
        context.context().list_physical_devices("GPU"))


# This is needed in case CUDA is lazily loaded.
app.call_after_init(_set_total_phsyical_gpus)


_TestResult = collections.namedtuple("_TestResult", ["status", "message"])


def _test_runner(test_id, test_env):
  # [FUNCTION DOC]
  # HELPER: entry point run inside worker processes by the multi-worker test wrapper.
  # KNOWN ISSUE: the docstring says it returns a boolean, but it actually returns a
  # result record holding a status string ("failure", "skipped" or "ok") and a message.
  # Inputs: test_id, the full test identifier; test_env, the test environment object from
  # the main process. Side effects: marks the process as a worker and replaces the
  # module-level environment with the received one; runs exactly one test with the standard
  # text runner; prints the first failure/error/skip message (traceback text) to standard
  # output. Failures, expected failures and errors all map to "failure" with the first
  # message; otherwise a skip maps to "skipped" with its reason; otherwise "ok" with no
  # message (unexpected successes count as ok).
  """Executes the test with the given test_id.

  This is a simple wrapper around TestRunner to be used with
  multi_process_runner. Similar to test.main(), but it executes only one test
  specified by test_id and returns whether the test succeeds. If the test fails,
  the function prints failures and errors to stdout.

  Args:
    test_id: TestCase.id()
    test_env: a TestEnvironment object.

  Returns:
    A boolean indicates whether the test succeeds.
  """
  global _running_in_worker, _env
  # No need to restore the value of _running_in_worker since it should always be
  # True in worker processes.
  _running_in_worker = True
  _env = test_env
  test = unittest.defaultTestLoader.loadTestsFromName(test_id)
  runner = unittest.TextTestRunner()
  result = runner.run(test)
  # Treat expected failures as failures, so that the main process can get
  # them and fail as expected. Also treat errors as failures to simplify the
  # handling.
  failures = result.failures + result.expectedFailures + result.errors
  if failures:
    ret = _TestResult(status="failure", message=failures[0][1])
  elif result.skipped:
    ret = _TestResult(status="skipped", message=result.skipped[0][1])
  else:
    # Treat unexpectedSuccesses as OK so that the test case in the main process
    # succeed as well.
    ret = _TestResult(status="ok", message=None)
  # Print tracebacks to stdout and multi_process_runner will collect
  # them and stream back to the main process.
  if ret.message:
    print(ret.message)
  return ret


def _multi_worker_test(test_method):
  # [FUNCTION DOC]
  # HELPER: wraps a test method so multi-worker combinations run it in worker processes.
  # Input: test_method, a test method function. Output: a decorated function (signature
  # preserved but extended with the five cluster parameters: chief flag, worker count,
  # parameter-server count, GPU sharing, runner), built with the decorator utility.
  # Behavior of the wrapper: in a worker process, for a single-process cluster, or for
  # parameter-server combinations under XLA, it enters a multi-worker session context and
  # calls the test directly. Otherwise (main process, multi-worker) it runs the test by id
  # in each worker, either via the supplied runner or a freshly created cluster and process
  # runner (started and joined), then inspects the results: the first failure fails the
  # test with its message; skipped results cause a skip only if no failure occurred.
  """Decorate test_method so that it runs in each worker.

  We use `multi_process_runner` to simulate multiple workers. Since we run the
  this function in the main process and all worker processes, this decoration
  behaves differently in the main process and worker procssses. In the main
  process, it spawns subprocesses and runs the test on each of them; in a worker
  process, it executes test in the same way as a normal test, e.g.
  setUp()/tearDown() are called before/after the test.

  Args:
    test_method: a function which must be a test method.

  Returns:
    Decorated `test_method`. Note that the decorated function has additional
    arguments.
  """

  def decorator(self, has_chief, num_workers, num_ps, share_gpu, runner,
                **kwargs):
    # [FUNCTION DOC]
    # HELPER: the wrapper installed by the multi-worker test decorator.
    # Inputs: the test case instance, the five cluster parameters, and the remaining test
    # arguments. Runs the test directly (inside a session suited to the strategy) when in a
    # worker, single-worker, or XLA parameter-server case; otherwise spawns the worker
    # processes, runs the same test in each, and fails or skips this test according to the
    # collected results (first failure wins; skip only when nothing failed). Returns nothing.
    if _num_total_workers(has_chief,
                          num_workers) == 1 or _running_in_worker or (
                              # Use in-process cluster for PS combinations
                              # when XLA is enabled.
                              test_util.is_xla_enabled() and num_ps > 0):
      # We're in worker process or the test is for single worker. Either case we
      # execute the test method directly instead of spawning subprocesses.

      # For MultiWorkerMirroredStrategy(CollectiveAllReduceStrategy), install a
      # session that connects to the local server. This is necessary for multi
      # worker graph mode tests to work. Those tests cannot use their graphs or
      # sessions, including the one returned by self.cached_session(). Since
      # existing tests may already be doing so, we only install the session for
      # multi worker tests.
      with _multi_worker_session(kwargs):
        test_method(self, **kwargs)
      return

    # We're in the main process. We spawn subprocesses and run the *test* on
    # each of them. Note that we're not directly executing test_method passed to
    # _multi_worker_test, because we need setUp()/tearDown() to be called and
    # all the decorations on the test method. The conceptual call stack is:
    #   [main process]test.main()
    #     [main process]test_runner.run(test)
    #       [main process]wrapper by combinations.generate()
    #         [main process]_multi_worker_test.decorator()
    #           # A sub process goes through the same code path as the main
    #           # process.
    #           [sub process]_test_runner()
    #             [sub process]test_runner.run(test)
    #               [sub process]wrapper by combinations.generate()
    #                 [sub process]_multi_worker_test.decorator()
    #                   # _running_in_worker is True
    #                   [sub process]test_method()
    test_id = self.id()
    if runner:
      results = runner.run(_test_runner, args=(test_id, _env))
    else:
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=num_workers,
          num_ps=num_ps,
          has_eval=False)
      ephemeral_runner = multi_process_runner.MultiProcessRunner(
          _test_runner,
          cluster_spec,
          share_gpu=share_gpu,
          args=(test_id, _env),
          dependence_on_chief=has_chief)
      ephemeral_runner.start()
      results = ephemeral_runner.join().return_value

    skip_reason = None
    for result in results:
      if result.status == "failure":
        # We can't tell which worker the return value come from, so we fail on
        # the  first error.
        self.fail(result.message)
        break
      elif result.status == "skipped":
        # Record the skip reason, but do not actually skip the test in case some
        # processes fail instead.
        skip_reason = result.message
    if skip_reason is not None:
      self.skipTest(skip_reason)

  argspec = tf_inspect.getfullargspec(test_method)
  decorator_args = (argspec.args or []) + [
      "has_chief", "num_workers", "num_ps", "share_gpu", "runner"
  ]
  decorator_argspec = argspec._replace(args=decorator_args)
  return tf_decorator.make_decorator(
      test_method, decorator, decorator_argspec=decorator_argspec)


def _num_total_workers(has_chief, num_workers):
  # [FUNCTION DOC]
  # HELPER: cluster-size arithmetic. Inputs: has_chief (boolean), num_workers (integer).
  # Returns the worker count plus one if a chief exists, otherwise the worker count.
  """Returns the number of workers including the chief."""
  if has_chief:
    return num_workers + 1
  return num_workers


def _multi_worker_session(kwargs):
  # [FUNCTION DOC]
  # HELPER: chooses the context to run a multi-worker test in.
  # Input: kwargs, mapping of test arguments. Scans for strategy objects; if more than one
  # is found it logs a warning and returns a no-op context manager. If the eager context is
  # active, or there is no collective all-reduce strategy, it also returns a no-op context.
  # Otherwise it copies the current context config, lets the strategy update it, targets the
  # strategy's cluster master, and returns a new session (as default-session context
  # manager) connected to it. Creating the session is a side effect.
  """Returns a context manager that enters a session that is configured for the MultiWorkerMirroredStrategy.

  Args:
    kwargs: a dict. Keyword arguments passed to the test.

  Returns:
    A context manager. If MultiWorkerMirroredStrategy is the  one and only one
    strategy in kwargs and it's in graph mode, it's the session that is
    configured for that strategy.  Otherwise, it's a no-op context manager.
  """
  strategy = None
  for _, v in kwargs.items():
    if isinstance(v, distribute_lib.StrategyBase):
      if strategy is not None:
        logging.warning(
            "The test uses multiple strategies. Skipping "
            "entering a session that is configured for the strategy.")
        return ops.NullContextmanager()
      strategy = v
  if context.executing_eagerly() or not isinstance(
      strategy, collective_all_reduce_strategy.CollectiveAllReduceStrategy):
    return ops.NullContextmanager()
  sess_config = copy.deepcopy(context.context().config)
  sess_config = strategy.update_config_proto(sess_config)
  target = strategy.cluster_resolver.master()
  return session.Session(config=sess_config, target=target).as_default()
