# [MODULE DOC]
# Facilities for generating parameterized test combinations. Exposes combine and times (build
# lists of ordered mappings forming cross products of options), generate (decorator expanding a
# test method or whole test class into one test per combination), TestCombination (hook class that
# can skip combinations, supply context managers and declare parameter modifiers),
# ParameterModifier and OptionalParameter (adjust or drop arguments before the test sees them) and
# NamedObject (wraps an object so it gets a readable test name). Public classes and functions are
# exported under the internal test combinations API namespace at import time via the export
# decorator; there are no other import-time side effects.
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
"""Facilities for creating multiple test combinations.

Here is a simple example for testing various optimizers in Eager and Graph:

class AdditionExample(test.TestCase, parameterized.TestCase):
  @combinations.generate(
     combinations.combine(mode=["graph", "eager"],
                          optimizer=[AdamOptimizer(),
                                     GradientDescentOptimizer()]))
  def testOptimizer(self, optimizer):
    ... f(optimizer)...

This will run `testOptimizer` 4 times with the specified optimizers: 2 in
Eager and 2 in Graph mode.
The test is going to accept the same parameters as the ones used in `combine()`.
The parameters need to match by name between the `combine()` call and the test
signature.  It is necessary to accept all parameters. See `OptionalParameter`
for a way to implement optional parameters.

`combine()` function is available for creating a cross product of various
options.  `times()` function exists for creating a product of N `combine()`-ed
results.

The execution of generated tests can be customized in a number of ways:
-  The test can be skipped if it is not running in the correct environment.
-  The arguments that are passed to the test can be additionally transformed.
-  The test can be run with specific Python context managers.
These behaviors can be customized by providing instances of `TestCombination` to
`generate()`.
"""

from collections import OrderedDict
import contextlib
import re
import types
import unittest

from absl.testing import parameterized

from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.test.combinations.TestCombination", v1=[])
class TestCombination:
  # [CLASS DOC]
  # Base class for customizing how generated test combinations run. Subclasses may override
  # should_execute_combination to skip a combination, parameter_modifiers to return modifier
  # objects that edit the test arguments, and context_managers to wrap the test in context
  # managers. The default implementations execute everything with no modifiers or managers.
  # Instances are passed to generate in a tuple; the class holds no state of its own.
  """Customize the behavior of `generate()` and the tests that it executes.

  Here is sequence of steps for executing a test combination:
    1. The test combination is evaluated for whether it should be executed in
       the given environment by calling `should_execute_combination`.
    2. If the test combination is going to be executed, then the arguments for
       all combined parameters are validated.  Some arguments can be handled in
       a special way.  This is achieved by implementing that logic in
       `ParameterModifier` instances that returned from `parameter_modifiers`.
    3. Before executing the test, `context_managers` are installed
       around it.
  """

  def should_execute_combination(self, kwargs):
    # [FUNCTION DOC]
    # Decides whether a combination should run in the current environment. Input kwargs is the
    # mapping of argument names to values for the combination (ignored by this default). Returns a
    # pair of a boolean and an optional reason string; the default always returns true with no
    # reason. No side effects and no exceptions.
    # KNOWN ISSUE: the docstring says None is returned instead of the string when the test runs,
    # and the caller later concatenates the reason onto a text prefix when skipping, so an override
    # returning false with no reason would raise a type error in the caller.
    """Indicates whether the combination of test arguments should be executed.

    If the environment doesn't satisfy the dependencies of the test
    combination, then it can be skipped.

    Args:
      kwargs:  Arguments that are passed to the test combination.

    Returns:
      A tuple boolean and an optional string.  The boolean False indicates
    that the test should be skipped.  The string would indicate a textual
    description of the reason.  If the test is going to be executed, then
    this method returns `None` instead of the string.
    """
    del kwargs
    return (True, None)

  def parameter_modifiers(self):
    # [FUNCTION DOC]
    # Returns the list of ParameterModifier instances this combination wants applied to the test
    # arguments. The default returns an empty list. Takes no inputs and has no side effects.
    """Returns `ParameterModifier` instances that customize the arguments."""
    return []

  def context_managers(self, kwargs):
    # [FUNCTION DOC]
    # Returns a list of already-created context managers under which the test is run. Input kwargs
    # is the mapping of argument names to values for the combination (ignored by default; a copy
    # is passed by the caller). The default returns an empty list. No side effects.
    """Return context managers for running the test combination.

    The test combination will run under all context managers that all
    `TestCombination` instances return.

    Args:
      kwargs:  Arguments and their values that are passed to the test
        combination.

    Returns:
      A list of instantiated context managers.
    """
    del kwargs
    return []


@tf_export("__internal__.test.combinations.ParameterModifier", v1=[])
class ParameterModifier:
  # [CLASS DOC]
  # Base class for objects that adjust the arguments passed to a generated test, for example
  # changing a value or removing an argument. Holds an optional parameter name. Two modifiers are
  # equal when they have the same exact type and name, and the hash uses the name (or the type
  # when no name is set), so duplicates collapse in a set. Defines a sentinel marking arguments
  # that must not be passed to the test.
  """Customizes the behavior of a particular parameter.

  Users should override `modified_arguments()` to modify the parameter they
  want, eg: change the value of certain parameter or filter it from the params
  passed to the test case.

  See the sample usage below, it will change any negative parameters to zero
  before it gets passed to test case.
  ```
  class NonNegativeParameterModifier(ParameterModifier):

    def modified_arguments(self, kwargs, requested_parameters):
      updates = {}
      for name, value in kwargs.items():
        if value < 0:
          updates[name] = 0
      return updates
  ```
  """

  DO_NOT_PASS_TO_THE_TEST = object()

  def __init__(self, parameter_name=None):
    # [FUNCTION DOC]
    # Stores the optional parameter name (a string or nothing) that this modifier is specific to.
    # Returns nothing; no exceptions.
    """Construct a parameter modifier that may be specific to a parameter.

    Args:
      parameter_name:  A `ParameterModifier` instance may operate on a class of
        parameters or on a parameter with a particular name.  Only
        `ParameterModifier` instances that are of a unique type or were
        initialized with a unique `parameter_name` will be executed.
        See `__eq__` and `__hash__`.
    """
    self._parameter_name = parameter_name

  def modified_arguments(self, kwargs, requested_parameters):
    # [FUNCTION DOC]
    # Computes updates to the test arguments. Inputs: kwargs is the mapping of combined argument
    # names to values; requested_parameters is the collection of parameter names in the test
    # signature. Returns a mapping of updates; a value equal to the do-not-pass sentinel means the
    # argument is removed. The default ignores both inputs and returns an empty mapping.
    """Replace user-provided arguments before they are passed to a test.

    This makes it possible to adjust user-provided arguments before passing
    them to the test method.

    Args:
      kwargs:  The combined arguments for the test.
      requested_parameters: The set of parameters that are defined in the
        signature of the test method.

    Returns:
      A dictionary with updates to `kwargs`.  Keys with values set to
      `ParameterModifier.DO_NOT_PASS_TO_THE_TEST` are going to be deleted and
      not passed to the test.
    """
    del kwargs, requested_parameters
    return {}

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality for modifiers: true for the same object, otherwise true only when both have exactly
    # the same type and the same stored parameter name. Input is any object; returns a boolean.
    """Compare `ParameterModifier` by type and `parameter_name`."""
    if self is other:
      return True
    elif type(self) is type(other):
      return self._parameter_name == other._parameter_name
    else:
      return False

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Inequality, defined as the logical negation of equality. Returns a boolean.
    return not self.__eq__(other)

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash consistent with equality: the hash of the parameter name when one is set, otherwise an
    # identifier derived from the class. Returns an integer.
    # KNOWN ISSUE: with no name the hash is per class, but with a name it ignores the class, so
    # equal-hash but unequal objects of different classes can collide (harmless but imprecise).
    """Compare `ParameterModifier` by type or `parameter_name`."""
    if self._parameter_name:
      return hash(self._parameter_name)
    else:
      return id(self.__class__)


@tf_export("__internal__.test.combinations.OptionalParameter", v1=[])
class OptionalParameter(ParameterModifier):
  # [CLASS DOC]
  # Modifier marking a parameter as optional in the combinations and in the test signature. If
  # the named parameter is not in the test signature it is dropped from the arguments, so a
  # TestCombination can consume it (for example a mode setting) without the test accepting it.
  """A parameter that is optional in `combine()` and in the test signature.

  `OptionalParameter` is usually used with `TestCombination` in the
  `parameter_modifiers()`. It allows `TestCombination` to skip certain
  parameters when passing them to `combine()`, since the `TestCombination` might
  consume the param and create some context based on the value it gets.

  See the sample usage below:

  ```
  class EagerGraphCombination(TestCombination):

    def context_managers(self, kwargs):
      mode = kwargs.pop("mode", None)
      if mode is None:
        return []
      elif mode == "eager":
        return [context.eager_mode()]
      elif mode == "graph":
        return [ops.Graph().as_default(), context.graph_mode()]
      else:
        raise ValueError(
            "'mode' has to be either 'eager' or 'graph', got {}".format(mode))

    def parameter_modifiers(self):
      return [test_combinations.OptionalParameter("mode")]
  ```

  When the test case is generated, the param "mode" will not be passed to the
  test method, since it is consumed by the `EagerGraphCombination`.
  """

  def modified_arguments(self, kwargs, requested_parameters):
    # [FUNCTION DOC]
    # Returns an empty mapping when the stored parameter name appears in the requested test
    # parameters, otherwise a mapping that removes that name via the do-not-pass sentinel. The
    # kwargs input is unused. No side effects.
    if self._parameter_name in requested_parameters:
      return {}
    else:
      return {self._parameter_name: ParameterModifier.DO_NOT_PASS_TO_THE_TEST}


def generate(combinations, test_combinations=()):
  # [FUNCTION DOC]
  # Decorator factory that expands a test method or an entire test class into one test per
  # combination. Inputs: combinations is a list of ordered mappings (as built by combine and
  # times); test_combinations is a tuple of TestCombination instances (default empty). Returns a
  # decorator. Applied to a class, the decorator replaces each method whose name starts with the
  # unittest test prefix by generated parameterized methods (mutating the class by deleting and
  # setting attributes and recording method ids) and returns the same class; applied to a function
  # it returns a named-parameterized version. Test names are derived from the combination keys
  # and values. Raises an assertion error if a combination is not an ordered mapping.
  # KNOWN ISSUE: the docstring says a value error is raised if parameters are not accepted; that
  # error is actually raised later, when a generated test runs, not when generate is applied.
  """A decorator for generating combinations of a test method or a test class.

  Parameters of the test method must match by name to get the corresponding
  value of the combination.  Tests must accept all parameters that are passed
  other than the ones that are `OptionalParameter`.

  Args:
    combinations: a list of dictionaries created using combine() and times().
    test_combinations: a tuple of `TestCombination` instances that customize
      the execution of generated tests.

  Returns:
    a decorator that will cause the test method or the test class to be run
    under the specified conditions.

  Raises:
    ValueError: if any parameters were not accepted by the test method
  """
  def decorator(test_method_or_class):
    # [FUNCTION DOC]
    # HELPER: inner decorator returned by generate.
    # Builds readable test names from the combinations (alphanumeric characters of keys and
    # values, with hexadecimal addresses replaced by the position), then wraps the given test
    # method or class. Returns the modified class or the parameterized method.
    """The decorator to be returned."""

    # Generate good test names that can be used with --test_filter.
    named_combinations = []
    for combination in combinations:
      # We use OrderedDicts in `combine()` and `times()` to ensure stable
      # order of keys in each dictionary.
      assert isinstance(combination, OrderedDict)
      name = "".join([
          "_{}_{}".format("".join(filter(str.isalnum, key)),
                          "".join(filter(str.isalnum, _get_name(value, i))))
          for i, (key, value) in enumerate(combination.items())
      ])
      named_combinations.append(
          OrderedDict(
              list(combination.items()) +
              [("testcase_name", "_test{}".format(name))]))

    if isinstance(test_method_or_class, type):
      class_object = test_method_or_class
      class_object._test_method_ids = test_method_ids = {}
      for name, test_method in class_object.__dict__.copy().items():
        if (name.startswith(unittest.TestLoader.testMethodPrefix) and
            isinstance(test_method, types.FunctionType)):
          delattr(class_object, name)
          methods = {}
          parameterized._update_class_dict_for_param_test_case(
              class_object.__name__, methods, test_method_ids, name,
              parameterized._ParameterizedTestIter(
                  _augment_with_special_arguments(
                      test_method, test_combinations=test_combinations),
                  named_combinations, parameterized._NAMED, name))
          for method_name, method in methods.items():
            setattr(class_object, method_name, method)

      return class_object
    else:
      test_method = _augment_with_special_arguments(
          test_method_or_class, test_combinations=test_combinations)
      return parameterized.named_parameters(*named_combinations)(test_method)

  return decorator


def _augment_with_special_arguments(test_method, test_combinations):
  # [FUNCTION DOC]
  # HELPER: internal glue for generate.
  # Wraps a test method so that, when a generated test runs, the test combinations can skip it,
  # modify its arguments and supply context managers. Inputs: the test method and a collection of
  # TestCombination instances. Returns the wrapper function.
  def decorated(self, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper produced by the argument augmentation.
    # Runs one generated test. Skips the test (skip exception) if any combination reports it should
    # not execute, collects parameter modifiers from all combinations, enters all context managers
    # (through an exit stack, or the legacy nested helper if present) and then runs the test.
    # Takes the test instance and the combination arguments as keyword arguments; returns nothing.
    # KNOWN ISSUE: the legacy branch using the nested context manager helper is Python 2 code and
    # is dead on Python 3.
    """A wrapped test method that can treat some arguments in a special way."""
    original_kwargs = kwargs.copy()

    # Skip combinations that are going to be executed in a different testing
    # environment.
    reasons_to_skip = []
    for combination in test_combinations:
      should_execute, reason = combination.should_execute_combination(
          original_kwargs.copy())
      if not should_execute:
        reasons_to_skip.append(" - " + reason)

    if reasons_to_skip:
      self.skipTest("\n".join(reasons_to_skip))

    customized_parameters = []
    for combination in test_combinations:
      customized_parameters.extend(combination.parameter_modifiers())
    customized_parameters = set(customized_parameters)

    # The function for running the test under the total set of
    # `context_managers`:
    def execute_test_method():
      # [FUNCTION DOC]
      # HELPER: runs the wrapped test under the installed context managers.
      # Applies each modifier's updates to the arguments (removing ones marked as not to be
      # passed), then checks the arguments against the test signature. Raises a value error if the
      # test requires parameters that were not passed, or if arguments were passed that the test
      # does not accept. Finally calls the test with the matching arguments including the instance.
      # Mutates the shared argument mapping.
      requested_parameters = tf_inspect.getfullargspec(test_method).args
      for customized_parameter in customized_parameters:
        for argument, value in customized_parameter.modified_arguments(
            original_kwargs.copy(), requested_parameters).items():
          if value is ParameterModifier.DO_NOT_PASS_TO_THE_TEST:
            kwargs.pop(argument, None)
          else:
            kwargs[argument] = value

      omitted_arguments = set(requested_parameters).difference(
          set(list(kwargs.keys()) + ["self"]))
      if omitted_arguments:
        raise ValueError("The test requires parameters whose arguments "
                         "were not passed: {} .".format(omitted_arguments))
      missing_arguments = set(list(kwargs.keys()) + ["self"]).difference(
          set(requested_parameters))
      if missing_arguments:
        raise ValueError("The test does not take parameters that were passed "
                         ": {} .".format(missing_arguments))

      kwargs_to_pass = {}
      for parameter in requested_parameters:
        if parameter == "self":
          kwargs_to_pass[parameter] = self
        else:
          kwargs_to_pass[parameter] = kwargs[parameter]
      test_method(**kwargs_to_pass)

    # Install `context_managers` before running the test:
    context_managers = []
    for combination in test_combinations:
      for manager in combination.context_managers(
          original_kwargs.copy()):
        context_managers.append(manager)

    if hasattr(contextlib, "nested"):  # Python 2
      # TODO(isaprykin): Switch to ExitStack when contextlib2 is available.
      with contextlib.nested(*context_managers):
        execute_test_method()
    else:  # Python 3
      with contextlib.ExitStack() as context_stack:
        for manager in context_managers:
          context_stack.enter_context(manager)
        execute_test_method()

  return decorated


@tf_export("__internal__.test.combinations.combine", v1=[])
def combine(**kwargs):
  # [FUNCTION DOC]
  # Builds the cross product of keyword options. Each keyword value is either a list of
  # possibilities or a single value treated as the only possibility. Returns a list of ordered
  # mappings with keys sorted alphabetically; with no arguments returns a list holding one empty
  # mapping. Implemented recursively; no side effects.
  """Generate combinations based on its keyword arguments.

  Two sets of returned combinations can be concatenated using +.  Their product
  can be computed using `times()`.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]`
         or `option=the_only_possibility`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
  if not kwargs:
    return [OrderedDict()]

  sort_by_key = lambda k: k[0]
  kwargs = OrderedDict(sorted(kwargs.items(), key=sort_by_key))
  first = list(kwargs.items())[0]

  rest = dict(list(kwargs.items())[1:])
  rest_combined = combine(**rest)

  key = first[0]
  values = first[1]
  if not isinstance(values, list):
    values = [values]

  return [
      OrderedDict(sorted(list(combined.items()) + [(key, v)], key=sort_by_key))
      for v in values
      for combined in rest_combined
  ]


@tf_export("__internal__.test.combinations.times", v1=[])
def times(*combined):
  # [FUNCTION DOC]
  # Computes the product of several combination lists, concatenating each pairing of mappings.
  # Input is one or more lists of mappings; with a single list it is returned unchanged. Returns a
  # list of ordered mappings. Raises a value error if paired mappings share any key, and an
  # assertion error if called with no arguments.
  """Generate a product of N sets of combinations.

  times(combine(a=[1,2]), combine(b=[3,4])) == combine(a=[1,2], b=[3,4])

  Args:
    *combined: N lists of dictionaries that specify combinations.

  Returns:
    a list of dictionaries for each combination.

  Raises:
    ValueError: if some of the inputs have overlapping keys.
  """
  assert combined

  if len(combined) == 1:
    return combined[0]

  first = combined[0]
  rest_combined = times(*combined[1:])

  combined_results = []
  for a in first:
    for b in rest_combined:
      if set(a.keys()).intersection(set(b.keys())):
        raise ValueError("Keys need to not overlap: {} vs {}".format(
            a.keys(), b.keys()))

      combined_results.append(OrderedDict(list(a.items()) + list(b.items())))
  return combined_results


@tf_export("__internal__.test.combinations.NamedObject", v1=[])
class NamedObject:
  # [CLASS DOC]
  # Transparent wrapper pairing an object with a name. Attribute access, calling and iteration are
  # forwarded to the wrapped object, while the text representation is the name, so the object
  # yields a readable test name when used as a combination value.
  """A class that translates an object into a good test name."""

  def __init__(self, name, obj):
    # [FUNCTION DOC]
    # Stores the display name (string) and the wrapped object. Returns nothing.
    self._name = name
    self._obj = obj

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Forwards lookup of any attribute not found on the wrapper to the wrapped object and returns
    # it; raises whatever the wrapped object raises for a missing attribute.
    return getattr(self._obj, name)

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Calls the wrapped object with all given positional and keyword arguments and returns its
    # result; exceptions propagate.
    return self._obj(*args, **kwargs)

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the wrapped object; raises if it is not iterable.
    return self._obj.__iter__()

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the stored name as the text representation.
    return self._name


def _get_name(value, index):
  # [FUNCTION DOC]
  # HELPER: builds test-name fragments for generate.
  # Converts a value to text and replaces any hexadecimal literal (such as a memory address) with
  # the given position index so names are stable between runs. Returns the resulting string.
  return re.sub("0[xX][0-9a-fA-F]+", str(index), str(value))
