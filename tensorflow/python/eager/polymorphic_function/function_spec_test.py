# [MODULE DOC]
# Unit tests for the FunctionSpec class of the function_type_utils module. The tests build
# FunctionSpec objects from small plain functions (optionally wrapped by a decorator-preserving
# wrapper or a transparent decorator) together with optional input signatures, and check the
# derived full argument specification, input signature, default values and FunctionType
# (parameter kinds, optionality and type constraints). Also covers the signature summary text
# and the legacy is_same_structure helper. No import-time side effects besides definitions.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for function_type_utils."""

from absl.testing import parameterized

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import tensor_spec
from tensorflow.python.platform import test
from tensorflow.python.util import tf_decorator


def dummy_tf_decorator(func):
  # [FUNCTION DOC]
  # HELPER: test decorator used to parameterize the FunctionSpec tests.
  # Wraps the given function in a generic pass-through wrapper and registers the wrapper with
  # the TensorFlow decorator utility so the original function stays discoverable as the target.
  # Input: func, the function to decorate. Output: the decorated wrapper function.
  # No side effects beyond creating the wrapper.

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: inner wrapper produced by the test decorator.
    # Forwards all positional and keyword arguments to the wrapped function and returns its
    # result unchanged.
    return func(*args, **kwargs)

  return tf_decorator.make_decorator(func, wrapper)


def transparent_decorator(func):
  # [FUNCTION DOC]
  # HELPER: identity decorator used to parameterize the FunctionSpec tests.
  # Returns the given function unchanged, serving as the "no decorator" case.
  return func


class FunctionSpecTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering FunctionSpec.from_function_and_signature for several function shapes:
  # required-only, optional-only, mixed, variadic, keyword-only, and bound/unbound methods.
  # Each test is parameterized over input signatures (none, full, partial, nested legacy list)
  # and over two decorators (a tf decorator wrapper and a transparent one).

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_required_only(self, input_signature, type_constraint, decorator):
    # [FUNCTION DOC]
    # Scenario: a function with three required positional parameters (x, y, z), decorated with
    # the given decorator. A FunctionSpec is built from it and the given input signature.
    # Asserts: the full argument spec lists the three names with no defaults, varargs, keyword
    # only args or annotations; input_signature equals what was passed; default values are
    # empty; and the function type has three positional-or-keyword, non-optional parameters
    # whose type constraints match the expected ones for each signature case.

    @decorator
    def foo(x, y, z):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: dummy target function with three required parameters; does nothing.
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y', 'z'], None, None, None, [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None),
               tensor_spec.TensorSpec(shape=None), trace_type.from_value(3))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None), trace_type.from_value(2),
               trace_type.from_value(3))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_optional_only(self, input_signature, type_constraint, decorator):
    # [FUNCTION DOC]
    # Scenario: a function whose three parameters all have defaults (1, 2, 3). A FunctionSpec is
    # built with the given input signature.
    # Asserts: the full argument spec carries the default tuple; the default values mapping
    # holds all three defaults; and the function type has three optional positional-or-keyword
    # parameters. Parameters not covered by a shorter signature get a type constraint derived
    # from their default value.

    @decorator
    def foo(x=1, y=2, z=3):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: dummy target function with three defaulted parameters; does nothing.
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y', 'z'], None, None, (1, 2, 3), [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'x': 1, 'y': 2, 'z': 3})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None),
               tensor_spec.TensorSpec(shape=None), trace_type.from_value(3))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_required_and_optional(self, input_signature, type_constraint,
                                 decorator):
    # [FUNCTION DOC]
    # Scenario: a function with two required parameters (x, y) and one defaulted parameter
    # (z with default 3). A FunctionSpec is built with the given input signature.
    # Asserts: the full argument spec has a single default; the default values mapping contains
    # only z; and the function type marks x and y as required and z as optional, with type
    # constraints matching the expected ones (z falls back to a constraint from its default
    # when the signature is shorter).

    @decorator
    def foo(x, y, z=3):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: dummy target function with two required and one defaulted parameter.
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y', 'z'], None, None, (3,), [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'z': 3})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_varargs(self, input_signature, type_constraint, decorator):
    # [FUNCTION DOC]
    # Scenario: a function taking only variadic positional arguments. A FunctionSpec is built
    # with an input signature of three entries.
    # Asserts: the variadic argument is expanded into three generated names (the base name with
    # indices 0 to 2) in the full argument spec, with no varargs entry kept; default values are
    # empty; and the function type has three positional-only, non-optional parameters with the
    # expected type constraints.

    @decorator
    def foo(*my_var_args):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: dummy target function with only variadic positional arguments.
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['my_var_args_0', 'my_var_args_1', 'my_var_args_2'
         ], None, None, None, [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'my_var_args_0', function_type_lib.Parameter.POSITIONAL_ONLY,
                False, type_constraint[0]),
            function_type_lib.Parameter(
                'my_var_args_1', function_type_lib.Parameter.POSITIONAL_ONLY,
                False, type_constraint[1]),
            function_type_lib.Parameter(
                'my_var_args_2', function_type_lib.Parameter.POSITIONAL_ONLY,
                False, type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None),
               tensor_spec.TensorSpec(shape=None), trace_type.from_value(3))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_kwonly(self, input_signature, type_constraint, decorator):
    # [FUNCTION DOC]
    # Scenario: a function with two required positional parameters and one keyword-only
    # parameter z with default 3. A FunctionSpec is built with the given input signature.
    # Asserts: the full argument spec lists x and y as arguments and z as keyword-only with its
    # default; default values contain only z; and the function type has x and y as
    # positional-or-keyword required parameters and z as an optional keyword-only parameter.

    @decorator
    def foo(x, y, *, z=3):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: dummy target function with a keyword-only defaulted parameter.
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec), (['x', 'y'], None, None, None, ['z'], {
            'z': 3
        }, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'z': 3})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.KEYWORD_ONLY, True,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              trace_type.from_value(1))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (None,
               trace_type.from_value([
                   tensor_spec.TensorSpec(shape=None),
                   tensor_spec.TensorSpec(shape=None)
               ], trace_type.InternalTracingContext(is_legacy_signature=True)),
               tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_method_bound_internal(
      self, input_signature, type_constraint, decorator
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the decorator parameter of this test is accepted but never used.
    # Scenario: a FunctionSpec is built from inside a decorator that is applied to a method
    # while the class body is being defined, so the function is still unbound and has the self
    # parameter. An instance method is then called to trigger the decorator's assertions.
    # Asserts (inside the decorator): the full argument spec includes self, x and y with the
    # default for y; default values contain only y; and the function type has self, x and y as
    # positional-or-keyword parameters where only y is optional, with the expected constraints
    # (self has no type constraint).

    def testing_decorator(func):
      # [FUNCTION DOC]
      # HELPER: decorator applied to the method inside the test class. Builds a FunctionSpec
      # from the raw function and the enclosing test's input signature, makes the assertions
      # described by the test, and returns the function unchanged.
      spec = function_type_utils.FunctionSpec.from_function_and_signature(
          func, input_signature
      )
      self.assertEqual(
          tuple(spec.fullargspec),
          (['self', 'x', 'y'], None, None, (1,), [], None, {}),
      )

      self.assertEqual(spec.default_values, {'y': 1})

      self.assertEqual(
          spec.function_type,
          function_type_lib.FunctionType([
              function_type_lib.Parameter(
                  'self',
                  function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                  False,
                  type_constraint[0],
              ),
              function_type_lib.Parameter(
                  'x',
                  function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                  False,
                  type_constraint[1],
              ),
              function_type_lib.Parameter(
                  'y',
                  function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                  True,
                  type_constraint[2],
              ),
          ]),
      )

      return func

    class MyClass:
      # [CLASS DOC]
      # HELPER: local class whose method is decorated by the testing decorator so the spec is
      # built from the unbound function at class-definition time.

      @testing_decorator
      def foo(self, x, y=1):
        # [FUNCTION DOC]
        # HELPER: dummy method with a required parameter and a defaulted one; does nothing.
        pass

    MyClass().foo(1)

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              trace_type.from_value(1))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_method_bound_external(
      self, input_signature, type_constraint, decorator
  ):
    # [FUNCTION DOC]
    # Scenario: a method is defined on a local class and a FunctionSpec is built from the method
    # bound to an instance, so self is already removed.
    # Asserts: the full argument spec lists only x and y with the default for y; default values
    # contain only y; and the function type has x (required) and y (optional) as
    # positional-or-keyword parameters with the expected type constraints.
    class MyClass:
      # [CLASS DOC]
      # HELPER: local class providing a decorated method for the bound-method scenario.

      @decorator
      def foo(self, x, y=1):
        # [FUNCTION DOC]
        # HELPER: dummy method with a required parameter and a defaulted one; does nothing.
        pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        MyClass().foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y'], None, None, (1,), [], None, {}),
    )
    self.assertEqual(spec.default_values, {'y': 1})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[1])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              trace_type.from_value(1))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (None,
               trace_type.from_value([
                   tensor_spec.TensorSpec(shape=None),
                   tensor_spec.TensorSpec(shape=None)
               ], trace_type.InternalTracingContext(is_legacy_signature=True)),
               tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_method_unbound(self, input_signature, type_constraint, decorator):
    # [FUNCTION DOC]
    # Scenario: a method is defined on a local class and a FunctionSpec is built from the
    # function accessed through the class (unbound), so self remains a parameter.
    # Asserts: the full argument spec lists self, x and y with the default for y; the input
    # signature is preserved; default values contain only y; and the function type has self (no
    # type constraint), x and y (optional) as positional-or-keyword parameters with the
    # expected constraints.

    class MyClass:
      # [CLASS DOC]
      # HELPER: local class providing a decorated method for the unbound-method scenario.

      @decorator
      def foo(self, x, y=1):
        # [FUNCTION DOC]
        # HELPER: dummy method with a required parameter and a defaulted one; does nothing.
        pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        MyClass.foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['self', 'x', 'y'], None, None, (1,), [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'y': 1})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'self', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                False, type_constraint[0]),
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[2])
        ]))

  def test_spec_summary(self):
    # [FUNCTION DOC]
    # Scenario: a decorated function with two defaulted parameters (2 and 3) and an input
    # signature of two unknown-shape tensor specs. Asserts that the verbose signature summary
    # text lists each parameter with its kind and tensor spec, then the output type, captures
    # (both reported as none) and the defaults section with both default values.
    input_signature = (
        tensor_spec.TensorSpec(shape=None),
        tensor_spec.TensorSpec(shape=None),
    )

    @dummy_tf_decorator
    def foo(x=2, y=3):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: dummy target function with two defaulted parameters; does nothing.
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature
    )
    self.assertEqual(
        spec.signature_summary(True),
        'Input Parameters:\n'
        + '  x (POSITIONAL_OR_KEYWORD):'
        ' TensorSpec(shape=<unknown>, dtype=tf.float32, name=None)\n'
        + '  y'
        ' (POSITIONAL_OR_KEYWORD): TensorSpec(shape=<unknown>,'
        ' dtype=tf.float32, name=None)\n'
        + 'Output Type:\n'
        + '  None\n'
        + 'Captures:\n'
        + '  None\n'
        + 'Defaults:\n'
        + '  x: 2\n'
        + '  y: 3',
    )


# TODO(fmuham): Remove when is_same_structure is removed.
class SameStructureTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the legacy is_same_structure helper in function_type_utils (slated for
  # removal per the TODO above the class).

  def test_same_structure(self):
    # [FUNCTION DOC]
    # Checks is_same_structure on flat lists. With the strict flag true, equal lists match and
    # lists differing in a value do not. With the flag false, only structure matters, so lists
    # of equal length match despite different values, while lists of different length do not.
    self.assertTrue(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 3], True)
    )
    self.assertTrue(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 4], False)
    )

    self.assertFalse(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 4], True)
    )
    self.assertFalse(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 3, 4], False)
    )


if __name__ == '__main__':
  test.main()
