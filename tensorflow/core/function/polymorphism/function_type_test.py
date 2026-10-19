# [MODULE DOC]
# Unit tests for the function_type module of the polymorphism package. They cover building
# function types from Python callables (required/optional/bound/unbound parameters), binding
# arguments with validation errors, pickling, canonicalizing call arguments into monomorphic
# types, the sub/supertype hierarchy, captures, argument-name sanitization, protobuf
# serialization and construction of a type from a structured signature.
# Run as a normal TensorFlow test file; has no import-time side effects beyond imports.
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
"""Tests for function_type."""

import collections
import pickle
import platform

from absl.testing import parameterized

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.framework import func_graph
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class FunctionTypeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for FunctionType construction from callables, default-value extraction, argument
  # binding validation and pickling. No shared fixtures.

  def test_required_only(self):
    # [FUNCTION DOC]
    # Builds a FunctionType from a function with three required positional-or-keyword
    # parameters and asserts it equals the hand-built type with non-optional parameters and no
    # type constraints; also asserts the default values mapping is empty.

    def foo(x, y, z):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Nested sample function with three required parameters; does nothing.
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None),
             function_type.Parameter(
                 "z", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None))))
    self.assertEqual(function_type.FunctionType.get_default_values(foo), {})

  def test_optional_only(self):
    # [FUNCTION DOC]
    # Builds a FunctionType from a function whose three parameters all have defaults; asserts
    # all parameters are marked optional and that the default values mapping holds 1, 2 and 3.

    def foo(x=1, y=2, z=3):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Nested sample function with three defaulted parameters; does nothing.
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, True, None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None),
             function_type.Parameter(
                 "z", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(foo), {
            "x": 1,
            "y": 2,
            "z": 3
        })

  def test_required_and_optional(self):
    # [FUNCTION DOC]
    # Builds a FunctionType from a function with two required and one defaulted parameter;
    # asserts the optional flags are false, false, true and that only the defaulted parameter
    # appears in the default values mapping.

    def foo(x, y, z=3):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Nested sample function with two required and one defaulted parameter; does nothing.
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None),
             function_type.Parameter(
                 "z", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(foo), {"z": 3})

  def test_method_bound(self):
    # [FUNCTION DOC]
    # Builds a FunctionType from a bound method; asserts the self parameter is dropped and the
    # remaining parameters (one required, one optional) are captured, with the default values
    # mapping holding only the defaulted one.

    class MyClass:
      # [CLASS DOC]
      # Sample class providing a method with one required and one defaulted parameter.

      def foo(self, x, y=1):
        # [FUNCTION DOC]
        # Sample method that does nothing.
        pass

    constraint = function_type.FunctionType.from_callable(MyClass().foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(MyClass().foo), {"y": 1})

  def test_method_unbound(self):
    # [FUNCTION DOC]
    # Builds a FunctionType from an unbound method accessed through the class; asserts the self
    # parameter is kept as a required first parameter, followed by x (required) and y
    # (optional), and that defaults contain only y.

    class MyClass:
      # [CLASS DOC]
      # Sample class providing a method with one required and one defaulted parameter.

      def foo(self, x, y=1):
        # [FUNCTION DOC]
        # Sample method that does nothing.
        pass

    constraint = function_type.FunctionType.from_callable(MyClass.foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "self", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(MyClass.foo), {"y": 1})

  def test_required_only_validation(self):
    # [FUNCTION DOC]
    # Checks argument binding on a type with two required parameters. Valid positional, keyword
    # and mixed combinations bind; asserts type errors for too many positional arguments, a
    # parameter given twice, an unexpected keyword and missing required arguments.

    def foo(x, y):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Nested sample function with two required parameters; does nothing.
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    constraint.bind(*(1, 2))
    constraint.bind(*(), **{"x": 1, "y": 2})
    constraint.bind(*(), **{"y": 1, "x": 2})
    constraint.bind(*(1,), **{"y": 2})

    with self.assertRaisesRegex(TypeError, "too many positional arguments"):
      constraint.bind(*(1, 2, 3))

    with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
      constraint.bind(*(1,), **{"x": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(1, 2), **{"z": 2})

    with self.assertRaisesRegex(TypeError, "missing a required argument: 'x'"):
      constraint.bind(*(), **{"z": 3})

    with self.assertRaisesRegex(TypeError, "missing a required argument: 'x'"):
      constraint.bind(*(), **{"y": 3})

  def test_optional_only_validation(self):
    # [FUNCTION DOC]
    # Checks argument binding on a type with two optional parameters. Omitting arguments is
    # allowed; asserts type errors for too many positional arguments, duplicated parameter
    # values and unexpected keywords.

    def foo(x=1, y=2):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Nested sample function with two defaulted parameters; does nothing.
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    constraint.bind(*(1, 2))
    constraint.bind(*(), **{"x": 1, "y": 2})
    constraint.bind(*(1,), **{"y": 2})
    constraint.bind(*(1,))
    constraint.bind(*(), **{"x": 1})

    with self.assertRaisesRegex(TypeError, "too many positional arguments"):
      constraint.bind(*(1, 2, 3))

    with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
      constraint.bind(*(1,), **{"x": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(1, 2), **{"z": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(), **{"z": 3})

  def test_required_and_optional_validation(self):
    # [FUNCTION DOC]
    # Checks argument binding on a type with one required and one optional parameter. Valid
    # combinations bind; asserts type errors for too many positional arguments, duplicated
    # values, unexpected keywords and a missing required argument.

    def foo(x, y=2):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Nested sample function with one required and one defaulted parameter; does nothing.
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    constraint.bind(*(1, 2))
    constraint.bind(*(), **{"x": 1, "y": 2})
    constraint.bind(*(1,), **{"y": 2})
    constraint.bind(*(1,))
    constraint.bind(*(), **{"x": 1})

    with self.assertRaisesRegex(TypeError, "too many positional arguments"):
      constraint.bind(*(1, 2, 3))

    with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
      constraint.bind(*(1,), **{"x": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(1, 2), **{"z": 2})

    with self.assertRaisesRegex(TypeError, "missing a required argument: 'x'"):
      constraint.bind(*(), **{"z": 3})

  def test_pickle(self):
    # [FUNCTION DOC]
    # Serializes a hand-built FunctionType with pickle, restores it and asserts the restored
    # object equals the original.
    original = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, None),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, None),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, None)
    ])
    cloned = pickle.loads(pickle.dumps(original))
    self.assertEqual(original, cloned)


class CanonicalizationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for canonicalizing call arguments into a monomorphic FunctionType and
  # binding them. Class attributes hold reusable argument/keyword-argument combinations
  # (positional only, mixed, keyword only, partial) used as parameter sets.
  args_1_2_3 = {"args": (1, 2, 3), "kwargs": {}}
  args_1_2_kwargs_z_3 = {"args": (1, 2), "kwargs": {"z": 3}}
  kwargs_x_1_y_2_z_3 = {"args": (), "kwargs": {"x": 1, "y": 2, "z": 3}}
  args_1_2 = {"args": (1, 2), "kwargs": {}}
  args_1_kwargs_y_2 = {"args": (1,), "kwargs": {"y": 2}}
  kwargs_x_1_y_2 = {"args": (), "kwargs": {"x": 1, "y": 2}}

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_required_only(self, args, kwargs):
    # [FUNCTION DOC]
    # Given a call (positional, mixed or keyword form) to a function with three required
    # parameters, canonicalizes it to a monomorphic type and asserts binding yields positional
    # arguments 1, 2, 3 with no keyword arguments, and that the type equals one with traced
    # constant types for each parameter.

    def foo(x, y, z):
      # [FUNCTION DOC]
      # Nested sample function with three required parameters; deletes them and does nothing.
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)
    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_optional_all(self, args, kwargs):
    # [FUNCTION DOC]
    # Same as the required-only case but for a function whose parameters all have defaults and
    # all are supplied by the call; asserts the bound arguments are 1, 2, 3 and the monomorphic
    # type has non-optional traced parameters.

    def foo(x=1, y=2, z=3):
      # [FUNCTION DOC]
      # Nested sample function with three defaulted parameters; does nothing.
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(args_1_2, args_1_kwargs_y_2, kwargs_x_1_y_2)
  def test_optional_some(self, args, kwargs):
    # [FUNCTION DOC]
    # Calls a function with defaults while omitting z; canonicalization receives the default
    # values mapping and fills z with 3. Mutates the kwargs argument by adding z, then asserts
    # binding gives positional 1, 2, 3 and the type equals the expected traced type.

    def foo(x=1, y=2, z=3):
      # [FUNCTION DOC]
      # Nested sample function with three defaulted parameters; does nothing.
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    default_values = function_type.FunctionType.get_default_values(foo)
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, default_values, {},
        polymorphic_type)

    kwargs["z"] = 3
    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_mixed(self, args, kwargs):
    # [FUNCTION DOC]
    # Canonicalizes calls to a function with two required and one defaulted parameter, with
    # the default not given a separate defaults mapping; asserts bound arguments are 1, 2, 3
    # and the resulting monomorphic type matches the expected traced parameters.

    def foo(x, y, z=3):
      # [FUNCTION DOC]
      # Nested sample function with two required and one defaulted parameter; does nothing.
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  def test_varargs(self):
    # [FUNCTION DOC]
    # Canonicalizes a call with three positional arguments to a variadic-positional function;
    # asserts the monomorphic type has positional-only parameters named after the variadic
    # name with index suffixes, and that binding returns the arguments unchanged.

    def foo(*my_var_args):
      # [FUNCTION DOC]
      # Nested sample variadic function; does nothing.
      del my_var_args

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    args = (1, 2, 3)
    kwargs = {}
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("my_var_args_0",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(1, type_context)),
        function_type.Parameter("my_var_args_1",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(2, type_context)),
        function_type.Parameter("my_var_args_2",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  def test_varkwargs(self):
    # [FUNCTION DOC]
    # Canonicalizes a call with keyword arguments x, y, z to a function accepting arbitrary
    # keywords; asserts the monomorphic type has keyword-only parameters with those names and
    # binding yields no positional arguments and the same keyword mapping.

    def foo(**kwargs):
      # [FUNCTION DOC]
      # Nested sample function taking arbitrary keywords; does nothing.
      del kwargs

    polymorphic_type = function_type.FunctionType.from_callable(foo)

    args = ()
    kwargs = {"x": 1, "y": 2, "z": 3}
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type
    )

    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, ())
    self.assertEqual(bound_args.kwargs, {"x": 1, "y": 2, "z": 3})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  def test_varargs_and_varkwargs(self):
    # [FUNCTION DOC]
    # Canonicalizes a call with one positional and two keyword arguments to a function taking
    # both variadics; asserts one positional-only parameter named with an index suffix, two
    # keyword-only parameters, and that binding reproduces the original arguments.

    def foo(*args, **kwargs):
      # [FUNCTION DOC]
      # Nested sample function taking arbitrary positionals and keywords; does nothing.
      del args, kwargs

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    args = (1,)
    kwargs = {"y": 2, "z": 3}
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type
    )

    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1,))
    self.assertEqual(bound_args.kwargs, {"y": 2, "z": 3})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("args_0",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(1, type_context)),
        function_type.Parameter("y", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_kwonly(self, args, kwargs):
    # [FUNCTION DOC]
    # Canonicalizes calls to a function with a keyword-only parameter z; asserts binding gives
    # positional arguments 1, 2 and keyword z=3, and the type marks z as keyword-only.

    def foo(x, y, *, z):
      # [FUNCTION DOC]
      # Nested sample function with a keyword-only parameter; does nothing.
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1, 2))
    self.assertEqual(bound_args.kwargs, {"z": 3})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
  )
  def test_posonly(self, args, kwargs):
    # [FUNCTION DOC]
    # Skipped unless the interpreter is Python 3.8 or newer. Builds, via dynamic evaluation of
    # source text, a function with two positional-only parameters followed by a normal one,
    # canonicalizes calls to it and asserts bound arguments 1, 2, 3 and that the first two
    # parameters are positional-only in the resulting type.
    major, minor, _ = platform.python_version_tuple()
    if not (major == "3" and int(minor) >= 8):
      self.skipTest("Positional only args are supported in Python 3.8+")

    # Raises syntax error in 3.7 but is important coverage for 3.8+.
    foo = eval("lambda x, y, /, z: x + y + z")  # pylint: disable=eval-used

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    bound_args = mono_type.bind(*args, **kwargs)
    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)


class TypeHierarchyTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the subtype/supertype relations and common-subtype computation of FunctionType,
  # plus placeholder argument generation.

  def test_same_type(self):
    # [FUNCTION DOC]
    # A type is equal to itself, is its own supertype and the most specific common subtype of
    # copies of itself is itself.
    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1))
    ])
    self.assertEqual(foo_type, foo_type)
    self.assertTrue(foo_type.is_supertype_of(foo_type))
    self.assertEqual(
        foo_type,
        foo_type.most_specific_common_subtype([foo_type, foo_type, foo_type]))

  def test_unrelated_types(self):
    # [FUNCTION DOC]
    # Two types with different constant constraints are unequal, neither is a supertype of the
    # other, and their common subtype computation returns nothing.
    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1))
    ])
    bar_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(2))
    ])
    self.assertNotEqual(foo_type, bar_type)
    self.assertFalse(foo_type.is_supertype_of(bar_type))
    self.assertIsNone(
        foo_type.most_specific_common_subtype([bar_type, bar_type]))
    self.assertIsNone(
        foo_type.most_specific_common_subtype([bar_type, foo_type]))

  def test_partial_raises_error(self):
    # [FUNCTION DOC]
    # Compares a fully constrained type with a partial one (missing type constraint); asserts
    # they are unequal and that supertype checks and common-subtype computation raise a type
    # error in both directions.
    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1)),
    ])
    bar_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, None)
    ])
    self.assertNotEqual(foo_type, bar_type)

    with self.assertRaises(TypeError):
      foo_type.is_supertype_of(bar_type)

    with self.assertRaises(TypeError):
      bar_type.is_supertype_of(foo_type)

    with self.assertRaises(TypeError):
      foo_type.most_specific_common_subtype([bar_type, bar_type])

    with self.assertRaises(TypeError):
      bar_type.most_specific_common_subtype([foo_type, bar_type])

  def test_related_types(self):
    # [FUNCTION DOC]
    # Uses two mock trace types (one always a supertype, one always a subtype) in single
    # parameter function types. Asserts the types differ, the subtype-based type is a supertype
    # check target of the other but not vice versa, and that the most specific common subtype of
    # both orders is the supertype-based function type.

    class MockAlwaysSuperType(trace.TraceType):
      # [CLASS DOC]
      # Mock trace type that is never a subtype of anything, whose common supertype is itself,
      # equality is identity and hash is constant zero.

      def is_subtype_of(self, other: trace.TraceType) -> bool:
        # [FUNCTION DOC]
        # Always returns false.
        return False

      def most_specific_common_supertype(self, others):
        # [FUNCTION DOC]
        # Returns this instance, ignoring the other types.
        return self

      def placeholder_value(self, placeholder_context):
        # [FUNCTION DOC]
        # Not supported; always raises a not-implemented error.
        raise NotImplementedError

      def __eq__(self, other):
        # [FUNCTION DOC]
        # Equality by object identity.
        return self is other

      def __hash__(self):
        # [FUNCTION DOC]
        # Returns the constant hash 0.
        return 0

    supertype = MockAlwaysSuperType()

    class MockAlwaysSubtype(trace.TraceType):
      # [CLASS DOC]
      # Mock trace type that is always a subtype of anything, whose common supertype is the
      # shared super mock instance, equality is identity and hash is constant one.

      def is_subtype_of(self, other) -> bool:
        # [FUNCTION DOC]
        # Always returns true.
        return True

      def most_specific_common_supertype(self, others):
        # [FUNCTION DOC]
        # Returns the enclosing test's super mock instance, ignoring the other types.
        return supertype

      def placeholder_value(self, placeholder_context):
        # [FUNCTION DOC]
        # Not supported; always raises a not-implemented error.
        raise NotImplementedError

      def __eq__(self, other):
        # [FUNCTION DOC]
        # Equality by object identity.
        return self is other

      def __hash__(self):
        # [FUNCTION DOC]
        # Returns the constant hash 1.
        return 1

    subtype = MockAlwaysSubtype()

    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, supertype),
    ])
    bar_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, subtype)
    ])

    self.assertNotEqual(foo_type, bar_type)
    self.assertTrue(bar_type.is_supertype_of(foo_type))
    self.assertFalse(foo_type.is_supertype_of(bar_type))

    self.assertEqual(
        foo_type.most_specific_common_subtype([bar_type, foo_type]), foo_type)
    self.assertEqual(
        bar_type.most_specific_common_subtype([bar_type, foo_type]), foo_type)

  def test_placeholder_arg(self):
    # [FUNCTION DOC]
    # Builds a type with a positional-only, a positional-or-keyword and a keyword-only
    # parameter constrained to constants 1, 2, 3, creates placeholders inside a function graph
    # and asserts the placeholder positional arguments are (1, 2) and keywords {z: 3}.
    type_context = trace_type.InternalTracingContext()
    foo = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])
    context_graph = func_graph.FuncGraph("test")
    placeholder_context = trace_type.InternalPlaceholderContext(context_graph)
    self.assertEqual(
        foo.placeholder_arguments(placeholder_context).args, (1, 2))
    self.assertEqual(
        foo.placeholder_arguments(placeholder_context).kwargs, {"z": 3})


class CapturesTest(test.TestCase):
  # [CLASS DOC]
  # Tests subtype and common-subtype behavior of FunctionType objects that differ in their
  # captures. Fixtures built in setUp are function types with captures a,b / a,b,c with various
  # constant values and a type capturing only d.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp and creates five parameterless function types with different
    # ordered capture mappings, stored as attributes of the test.
    super(CapturesTest, self).setUp()

    def gen_type_fn(mapping):
      # [FUNCTION DOC]
      # HELPER: builds the capture fixtures in setUp.
      # Creates a function type without parameters whose captures are an ordered copy of the
      # given mapping from names to trace types.
      return function_type.FunctionType([], collections.OrderedDict(mapping))

    self.type_a1_b1 = gen_type_fn({
        "a": trace_type.from_value(1),
        "b": trace_type.from_value(1)
    })
    self.type_a1_b1_c1 = gen_type_fn({
        "a": trace_type.from_value(1),
        "b": trace_type.from_value(1),
        "c": trace_type.from_value(1)
    })
    self.type_a2_b2_c2 = gen_type_fn({
        "a": trace_type.from_value(2),
        "b": trace_type.from_value(2),
        "c": trace_type.from_value(2)
    })
    self.type_a1_b1_c2 = gen_type_fn({
        "a": trace_type.from_value(1),
        "b": trace_type.from_value(1),
        "c": trace_type.from_value(2)
    })
    self.type_d1 = gen_type_fn({"d": trace_type.from_value(1)})

  def testCapturesSubtype(self):
    # [FUNCTION DOC]
    # Asserts supertype relations: fewer captures with equal values is a supertype of more
    # captures; the reverse, differing values and disjoint captures are not.
    self.assertTrue(self.type_a1_b1.is_supertype_of(self.type_a1_b1_c1))
    self.assertFalse(self.type_a1_b1_c1.is_supertype_of(self.type_a1_b1))
    self.assertFalse(self.type_a1_b1_c1.is_supertype_of(self.type_a2_b2_c2))
    self.assertFalse(self.type_a1_b1_c1.is_supertype_of(self.type_a2_b2_c2))
    self.assertFalse(self.type_d1.is_supertype_of(self.type_a1_b1))

  def testCapturesSupertype(self):
    # [FUNCTION DOC]
    # Checks common-subtype results over captures: the result merges capture sets (3 or 4
    # captures) when values agree and is nothing when values conflict.
    supertype_1 = self.type_a1_b1_c1.most_specific_common_subtype(
        [self.type_a1_b1_c1])
    self.assertLen(supertype_1.captures, 3)

    supertype_2 = self.type_a1_b1.most_specific_common_subtype(
        [self.type_a1_b1_c1, self.type_a2_b2_c2])
    self.assertIsNone(supertype_2)

    supertype_3 = self.type_a1_b1.most_specific_common_subtype(
        [self.type_a1_b1_c2])
    self.assertLen(supertype_3.captures, 3)

    supertype_4 = self.type_a1_b1_c1.most_specific_common_subtype(
        [self.type_a1_b1_c2])
    self.assertIsNone(supertype_4)

    supertype_5 = self.type_a1_b1_c1.most_specific_common_subtype(
        [self.type_d1])
    self.assertLen(supertype_5.captures, 4)


class SanitizationTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the argument-name sanitization function.

  def testRename(self):
    # [FUNCTION DOC]
    # Asserts that names starting with a digit or underscore get an arg prefix, invalid
    # characters become underscores, and already valid names are unchanged.
    self.assertEqual("arg_42", function_type.sanitize_arg_name("42"))
    self.assertEqual("a42", function_type.sanitize_arg_name("a42"))
    self.assertEqual("arg__42", function_type.sanitize_arg_name("_42"))
    self.assertEqual("a___", function_type.sanitize_arg_name("a%$#"))
    self.assertEqual("arg____", function_type.sanitize_arg_name("%$#"))
    self.assertEqual("foo", function_type.sanitize_arg_name("foo"))
    self.assertEqual("Foo", function_type.sanitize_arg_name("Foo"))
    self.assertEqual("arg_96ab_cd___53",
                     function_type.sanitize_arg_name("96ab.cd//?53"))

  def testLogWarning(self):
    # [FUNCTION DOC]
    # Sanitizes an invalid name while capturing warning-level logs; asserts the sanitized
    # result and that a warning stating the original and sanitized names was logged.

    with self.assertLogs(level="WARNING") as logs:
      result = function_type.sanitize_arg_name("96ab.cd//?53")

    self.assertEqual(result, "arg_96ab_cd___53")

    expected_message = (
        "WARNING:absl:`96ab.cd//?53` is not a valid tf.function parameter name."
        " Sanitizing to `arg_96ab_cd___53`.")
    self.assertIn(expected_message, logs.output)


class SerializationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests protobuf round-tripping of Parameter and FunctionType and their string form.

  @parameterized.product(
      name=["arg_0", "param"],
      kind=[
          function_type.Parameter.POSITIONAL_ONLY,
          function_type.Parameter.POSITIONAL_OR_KEYWORD
      ],
      optional=[True, False],
      type_contraint=[None, trace_type.from_value(1)])
  def testParameter(self, name, kind, optional, type_contraint):
    # [FUNCTION DOC]
    # Over the product of names, kinds, optional flag and an optional constant type constraint,
    # asserts that a Parameter converts to the expected protobuf message (with serialized
    # constraint when present) and that parsing the message returns an equal Parameter.
    original = function_type.Parameter(name, kind, optional, type_contraint)
    expected_type_constraint = serialization.serialize(
        type_contraint) if type_contraint else None
    expected = function_type_pb2.Parameter(
        name=name,
        kind=function_type.PY_TO_PROTO_ENUM[kind],
        is_optional=optional,
        type_constraint=expected_type_constraint)
    self.assertEqual(original.to_proto(), expected)
    self.assertEqual(function_type.Parameter.from_proto(expected), original)

  def testFunctionType(self):
    # [FUNCTION DOC]
    # Asserts a FunctionType with one parameter and one capture converts to the expected
    # protobuf message and that parsing it back gives an equal type.
    original = function_type.FunctionType([
        function_type.Parameter("a", function_type.Parameter.POSITIONAL_ONLY,
                                False, None),
    ], collections.OrderedDict([("b", trace_type.from_value(1))]))
    expected = function_type_pb2.FunctionType(
        parameters=[
            function_type_pb2.Parameter(
                name="a",
                kind=function_type_pb2.Parameter.Kind.POSITIONAL_ONLY,
                is_optional=False)
        ],
        captures=[
            function_type_pb2.Capture(
                name="b",
                type_constraint=serialization.serialize(
                    trace_type.from_value(1)))
        ])
    self.assertEqual(original.to_proto(), expected)
    self.assertEqual(function_type.FunctionType.from_proto(expected), original)

  def testCapturedDefaultValueStr(self):
    # [FUNCTION DOC]
    # Asserts the string form of a type with one optional unconstrained parameter shows the
    # placeholder text for a captured default value.
    f_type = function_type.FunctionType([
        function_type.Parameter(
            "a", function_type.Parameter.POSITIONAL_OR_KEYWORD, True, None
        ),
    ])

    self.assertEqual(str(f_type), "(a=<captured_default_value>)")


class FromStructuredSignatureTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for building a function type from a structured signature.

  @parameterized.parameters(
      {
          "signature": ((1, 2, 3), {}),
          "expected_types": (
              trace_type.from_value(1),
              trace_type.from_value(2),
              trace_type.from_value(3),
          ),
      },
      {
          "signature": (([1, 2, 3],), {}),
          "expected_types": (
              trace_type.from_value([1, 2, 3]),
          ),
      },
      {
          "signature": ((), {}),
          "expected_types": (),
      },
  )
  def testArgs(self, signature, expected_types):
    # [FUNCTION DOC]
    # For signatures with only positional arguments, asserts the output type is that of a
    # null value, every parameter is positional-only and each constraint equals the expected
    # traced type.
    generated_type = function_type.from_structured_signature(signature)
    self.assertEqual(generated_type.output, trace_type.from_value(None))
    for i, p in enumerate(generated_type.parameters.values()):
      self.assertEqual(p.kind, function_type.Parameter.POSITIONAL_ONLY)
      self.assertEqual(p.type_constraint, expected_types[i])

  @parameterized.parameters(
      {
          "signature": ((), {"a": 1, "b": 2, "c": 3}),
          "expected_types": {
              "a": trace_type.from_value(1),
              "b": trace_type.from_value(2),
              "c": trace_type.from_value(3),
          },
      },
      {
          "signature": ((), {"a": [1, 2, 3]}),
          "expected_types": {
              "a": trace_type.from_value([1, 2, 3]),
          },
      },
      {
          "signature": ((), {}),
          "expected_types": {},
      },
  )
  def testKwargs(self, signature, expected_types):
    # [FUNCTION DOC]
    # For signatures with only keyword arguments, asserts the output type is that of a null
    # value, every parameter is keyword-only and each constraint equals the expected type for
    # its name.
    generated_type = function_type.from_structured_signature(signature)
    self.assertEqual(generated_type.output, trace_type.from_value(None))
    for p in generated_type.parameters.values():
      self.assertEqual(p.kind, function_type.Parameter.KEYWORD_ONLY)
      self.assertEqual(p.type_constraint, expected_types[p.name])

  @parameterized.parameters(
      {"output_signature": 1},
      {"output_signature": [1, 2, 3]},
      {"output_signature": ()},
  )
  def testOutput(self, output_signature):
    # [FUNCTION DOC]
    # Builds a type from an empty input signature and the given output signature and asserts
    # the output type equals tracing the output value in a legacy-signature context.
    generated_type = function_type.from_structured_signature(
        ((), {}), output_signature
    )
    self.assertEqual(
        generated_type.output,
        trace_type.from_value(
            output_signature,
            trace_type.InternalTracingContext(is_legacy_signature=True),
        )
    )

if __name__ == "__main__":
  test.main()
