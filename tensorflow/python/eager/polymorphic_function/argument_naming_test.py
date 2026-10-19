# [MODULE DOC]
# Tests that concrete functions obtained from polymorphic (traced) functions get recognizable,
# stable input names, as needed for exported signatures. Covers plain arguments, variables as
# arguments, defaults and dictionary outputs, user-given tensor-spec names overriding argument
# names, methods on classes, input signatures on methods, and variadic positional and keyword
# arguments. Each test checks both operation names and the user-specified-name attribute of the
# concrete function's inputs. Running as a script enables eager execution with four CPU devices
# and starts the test main. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ArgumentNamingTests(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case verifying how argument names (from the Python signature or from tensor-spec
  # names) become input names of concrete functions. Parameterized base is present but no
  # parameterized tests are defined.
  """Tests for recognizable export signatures from concrete functions."""

  def testBasic(self):
    # [FUNCTION DOC]
    # Test: a traced function of two arguments returning their sum and product. After a warm-up
    # call, a concrete function is requested with specs for both arguments; asserts the input
    # operations are named a and b, carry matching user-specified-name attributes, there are two
    # structured outputs, and calling it positionally or by keyword gives sum 3 and product 2.
    @polymorphic_function.function
    def fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a pair: a plus b and a times b.
      return a + b, a * b
    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['a', 'b'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertLen(fn_op.graph.structured_outputs, 2)
    self.assertAllClose(
        [3., 2.],
        fn_op(constant_op.constant(1.), constant_op.constant(2.)))
    self.assertAllClose(
        [3., 2.],
        fn_op(a=constant_op.constant(1.), b=constant_op.constant(2.)))

  def testVariable(self):
    # [FUNCTION DOC]
    # Test: like the basic test but the second argument is given as a variable instead of a
    # spec. Asserts the input operations are still named a and b (with matching user-specified
    # names) and there are two structured outputs.
    @polymorphic_function.function
    def fn(a, b):
      # [FUNCTION DOC]
      # Traced function returning a pair: a plus b and a times b.
      return a + b, a * b
    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        variables.Variable(1.))
    self.assertEqual(
        ['a', 'b'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertLen(fn_op.graph.structured_outputs, 2)

  def testDictReturned(self):
    # [FUNCTION DOC]
    # Test: a traced function with a default pair argument and default scalar returning a
    # dictionary. Checks three concrete functions: (1) only x and y given as specs: inputs named
    # x and y and structured outputs have keys alpha and beta; (2) pair z given with named specs,
    # y with a custom name, x as a Python number: inputs named z_first, z_second, custom;
    # (3) first positional spec named custom plus z with named specs and y unnamed: inputs named
    # custom, z1, z2, y. Each time operation names and user-specified-name attributes are
    # compared.
    @polymorphic_function.function
    def fn(x, z=(1., 2.), y=3.):
      # [FUNCTION DOC]
      # Traced function: x is a tensor, z is a pair of numbers or tensors (default one and two),
      # y a scalar (default three). Returns a mapping with alpha equal to x plus y plus the first
      # element of z, and beta equal to x times y plus the second element of z.
      z1, z2 = z
      return {'alpha': x + y + z1, 'beta': x * y + z2}
    # Call the function to make def_function happy
    fn(array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        x=tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x', 'y'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'x', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertEqual({'alpha', 'beta'},
                     set(fn_op.graph.structured_outputs.keys()))

    fn_op2 = fn.get_concrete_function(
        z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32,
                                  name='z_first'),
           tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32,
                                  name='z_second')),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        x=4.)
    self.assertEqual(
        ['z_first', 'z_second', 'custom'],
        [inp.op.name for inp in fn_op2.inputs])
    self.assertEqual(
        [b'z_first', b'z_second', b'custom'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op2.inputs])

    fn_op3 = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32,
                                  name='z1'),
           tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z2')),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['custom', 'z1', 'z2', 'y'],
        [inp.op.name for inp in fn_op3.inputs])
    self.assertEqual(
        [b'custom', b'z1', b'z2', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op3.inputs])

  def testMethod(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test calls the skip function in the middle of the body, so the final
    # block (overriding the input name through a spec name on a bound method) never runs; it is
    # dead code marked as not working (a TODO notes that spec names may need to be part of cache
    # keys).
    # Test: a class with a traced method. Concrete functions requested via the class (passing
    # the instance explicitly) and via the bound method both must have a single input named x
    # (operation name and user-specified-name attribute). Then the test is skipped.
    class HasMethod(object):
      # [CLASS DOC]
      # Local helper class with a single traced method that returns its argument.

      @polymorphic_function.function
      def method(self, x):
        # [FUNCTION DOC]
        # Traced method returning x unchanged.
        return x

    has_method = HasMethod()
    # Call the function to make def_function happy
    HasMethod.method(has_method, array_ops.ones([]))
    class_op = HasMethod.method.get_concrete_function(
        has_method, tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x'],
        [inp.op.name for inp in class_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in class_op.inputs])
    # Call the function to make def_function happy
    has_method.method(array_ops.ones([]))
    method_op = has_method.method.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
    # TODO(allenl): It should be possible to override names when exporting. Do
    # TensorSpec names need to go in cache keys? Or maybe get_concrete_function
    # should always retrace?
    self.skipTest('Not working')
    method_op = has_method.method.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='y'))
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])

  def testMethodSignature(self):
    # [FUNCTION DOC]
    # Test: a traced method with a fixed input signature whose spec is named y. After a warm-up
    # call, concrete functions requested twice with no arguments must both have one input named
    # y (operation name and user-specified-name attribute), showing the spec name wins over the
    # argument name x.

    class HasMethod(object):
      # [CLASS DOC]
      # Local helper class whose method has an input signature of one unknown-shape double
      # tensor named y.

      @polymorphic_function.function(
          input_signature=(tensor_spec.TensorSpec(
              shape=None, dtype=dtypes.float64, name='y'),))
      def method(self, x):
        # [FUNCTION DOC]
        # Traced method that checks the instance can be hashed (no weak proxy is passed as the
        # instance) and returns x unchanged.
        hash(self)  # No weak proxies passed as `self`
        return x

    has_method = HasMethod()
    # Call the function to make def_function happy
    has_method.method(array_ops.ones([], dtype=dtypes.float64))
    method_op = has_method.method.get_concrete_function()
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
    method_op2 = has_method.method.get_concrete_function()
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op2.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op2.inputs])

  def testVariadic(self):
    # [FUNCTION DOC]
    # Test: a traced function with a positional argument plus variadic positional and keyword
    # arguments. Requests a concrete function with specs for x, two unnamed or named positional
    # extras and two keyword extras (one custom-named). Asserts input names are x, y, args_1,
    # second_variadic, z, cust: spec names win, unnamed variadic items are numbered by position,
    # unnamed keyword items use the keyword. User-specified-name attributes are the same.
    @polymorphic_function.function
    def variadic_fn(x, *args, **kwargs):
      # [FUNCTION DOC]
      # Traced function returning x plus the sum of all extra positional and keyword arguments.
      return x + math_ops.add_n(list(args) + list(kwargs.values()))

    # Call the function to make def_function happy
    variadic_fn(array_ops.ones([]), array_ops.ones([]))
    variadic_op = variadic_fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32,
                               name='second_variadic'),
        z=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        zz=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='cust'))
    self.assertEqual(
        ['x', 'y', 'args_1', 'second_variadic', 'z', 'cust'],
        [inp.op.name for inp in variadic_op.inputs])
    self.assertEqual(
        [b'x', b'y', b'args_1', b'second_variadic', b'z', b'cust'],
        [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs])

  def testVariadicInputSignature(self):
    # [FUNCTION DOC]
    # Test: a traced variadic function with a four-spec input signature (two named y and z).
    # After a warm-up call with four tensors, the concrete function's name contains the function
    # name, and its inputs are named x, y, args_1, z (spec names win, unnamed variadic items are
    # numbered), with matching user-specified-name attributes.
    @polymorphic_function.function(
        input_signature=(
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z'),
        ))
    def variadic_fn(x, *args):
      # [FUNCTION DOC]
      # Traced function returning x plus the sum of all extra positional arguments.
      return x + math_ops.add_n(list(args))

    # Call the function to make def_function happy
    variadic_fn(array_ops.ones([]), array_ops.ones([]),
                array_ops.ones([]), array_ops.ones([]))
    variadic_op = variadic_fn.get_concrete_function()
    self.assertIn(b'variadic_fn', variadic_op.name)
    self.assertEqual(
        ['x', 'y', 'args_1', 'z'],
        [inp.op.name for inp in variadic_op.inputs])
    self.assertEqual(
        [b'x', b'y', b'args_1', b'z'],
        [inp.op.get_attr('_user_specified_name')
         for inp in variadic_op.inputs])


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={'CPU': 4}))
  test.main()
