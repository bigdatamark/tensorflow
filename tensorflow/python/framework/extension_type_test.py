# [MODULE DOC]
# Large test suite for user-defined extension types (typed composite objects whose fields may
# be tensors, ragged tensors, tuples, mappings and python values). Defines several example
# extension types used as fixtures (masked tensors, forward-reference types, types with tensor
# defaults) plus test classes covering attribute access, immutability, constructors, equality,
# use in compiled functions, control flow, datasets, batching, type specifications, dispatch,
# pickling and saved-model round trips. Helper functions build placeholders and compare values.
# No import-time side effects beyond defining the fixture types, which register their classes.
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
"""Tests for tf.framework.extension_type."""

import contextlib
import copy
import pickle
import tempfile
import typing

from absl.testing import parameterized
import typing_extensions

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


POSITIONAL_OR_KEYWORD = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = tf_inspect.Parameter.KEYWORD_ONLY


class MaskedTensorV1(extension_type.ExtensionType):
  # [CLASS DOC]
  # Minimal fixture extension type with two tensor fields, values and mask. Has no custom
  # methods, so it exercises the default constructor, printed form and equality behavior.
  """Example subclass of ExtensionType, used for testing."""

  values: tensor.Tensor
  mask: tensor.Tensor


class MaskedTensorV2(extension_type.ExtensionType):
  # [CLASS DOC]
  # Richer fixture extension type with tensor fields values and mask and a registered name.
  # Adds a custom printed form, shape and dtype properties, an alternate constructor from a
  # full tensor, a static method, a validation hook, a default-filling method and the
  # addition and subtraction operators (bound to the math add and subtract ops).
  """Example subclass of ExtensionType, used for testing.

  This version adds methods, classmethod, staticmethod, and properties, and
  customizes `__repr__` and `__validate__`.  It also adds a `__name__` field,
  which enables serialization.
  """

  __name__ = 'tf.test.MaskedTensorV2'

  values: tensor.Tensor
  mask: tensor.Tensor

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a printable form. When both fields are eager tensors (expose numpy data) it
    # renders a masked-array style text with masked-out items shown as underscores; otherwise
    # (symbolic tensors) it defers to the base class printed form. No side effects.
    if hasattr(self.values, 'numpy') and hasattr(self.mask, 'numpy'):
      return '<MaskedTensorV2 %s>' % _masked_array_repr(
          self.values.numpy(), self.mask.numpy()
      )
    else:
      return super(MaskedTensorV2, self).__repr__()

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property returning the static shape of the values tensor.
    return self.values.shape

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the element type of the values tensor.
    return self.values.dtype

  @classmethod
  def from_full_tensor(cls, values):
    # [FUNCTION DOC]
    # Alternate constructor. Input: values, a tensor-like object. Returns an instance whose
    # mask is all true (same shape as values), i.e. nothing is masked out.
    return cls(values, array_ops.ones_like(values, dtype=dtypes.bool))

  # A dummy example to test support of staticmethod
  @staticmethod
  def doc_link():
    # [FUNCTION DOC]
    # Static method with no inputs; returns a fixed example documentation address string.
    # Exists to test that static methods are supported on extension types.
    return 'http://example.com/masked_tensor'

  def __validate__(self):
    # [FUNCTION DOC]
    # Validation hook run after construction. Checks that the values and mask shapes are
    # compatible and raises a value error if they are not. Returns nothing.
    self.values.shape.assert_is_compatible_with(self.mask.shape)

  def with_default(self, default):
    # [FUNCTION DOC]
    # Input: default, a value or tensor broadcastable with values. Returns a tensor holding the
    # value where the mask is true and the default elsewhere.
    return array_ops.where_v2(self.mask, self.values, default)

  __add__ = math_ops.add
  __sub__ = math_ops.subtract


class SimpleExtensionType(extension_type.ExtensionType):
  # [CLASS DOC]
  # Fixture extension type with a single tensor field x, which customizes its nested type
  # specification class by adding a shape property.
  x: tensor.Tensor

  class Spec:
    # [CLASS DOC]
    # Customization of the generated type specification: its members are mixed into the
    # automatically created spec class for this type.

    @property
    def shape(self):
      # [FUNCTION DOC]
      # Property on the type specification returning the shape of the x field's spec.
      return self.x.shape


def _masked_array_repr(values, mask):
  # [FUNCTION DOC]
  # HELPER: builds the printed text for the masked tensor fixture's eager printed form.
  # Inputs: values and mask, numpy arrays of equal leading length (asserted). Recursively
  # renders a bracketed comma separated list; for one-dimensional input, items whose mask is
  # true show the element's printed value and others show an underscore. Returns the string.
  """Returns a string representation for a masked numpy array."""
  assert len(values) == len(mask)
  if len(values.shape) == 1:
    items = [repr(v.item()) if m else '_' for (v, m) in zip(values, mask)]
  else:
    items = [_masked_array_repr(v, m) for (v, m) in zip(values, mask)]
  return '[%s]' % ', '.join(items)


class MaskedTensorV3(extension_type.BatchableExtensionType):
  # [CLASS DOC]
  # Batchable fixture extension type whose fields may be dense or ragged tensors. Has a custom
  # constructor, shape and dtype properties, and a custom spec class exposing a private shape,
  # to emulate what the Keras integration requires of tensor-like objects.
  """Example subclass of ExtensionType, used for testing.

  This version adds Keras required properties to MaskedTensor and its Spec
  class, to test Keras integration.
  """

  __name__ = 'tf.test.MaskedTensorV3.Spec'

  values: typing.Union[tensor.Tensor, ragged_tensor.RaggedTensor]
  mask: typing.Union[tensor.Tensor, ragged_tensor.RaggedTensor]

  def __init__(self, values, mask):
    # [FUNCTION DOC]
    # Custom constructor. Inputs: values and mask. If values is a ragged tensor the mask must
    # also be ragged with boolean dtype (asserted); otherwise both are converted to dense
    # tensors, the mask as boolean. Stores both as fields. Returns nothing.
    if isinstance(values, ragged_tensor.RaggedTensor):
      assert isinstance(mask, ragged_tensor.RaggedTensor)
      assert mask.dtype == dtypes.bool
    else:
      values = ops.convert_to_tensor(values)
      mask = ops.convert_to_tensor(mask, dtypes.bool)
    self.values = values
    self.mask = mask

  # Required by assert_input_compatibility in keras/engine/input_spec.py
  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property returning the static shape of the values field (needed by Keras input checks).
    return self.values.shape

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the element type of the values field.
    return self.values.dtype

  class Spec:
    # [CLASS DOC]
    # Customization of the generated type specification; adds the private shape property that
    # Keras tensors read.

    # Required by KerasTensor.shape in keras/engine/keras_tensor.py
    @property
    def _shape(self):
      # [FUNCTION DOC]
      # Property on the type specification returning the private shape of the values field spec.
      return self.values._shape


class ForwardRefA(extension_type.ExtensionType):
  # [CLASS DOC]
  # Fixture type whose field annotations refer by name to itself and to ForwardRefB, which is
  # defined afterwards; used to test resolution of forward references. Fields: a tuple of
  # either type, and a ForwardRefB.
  x: typing.Tuple[typing.Union['ForwardRefA', 'ForwardRefB'], ...]
  y: 'ForwardRefB'


class ForwardRefB(extension_type.ExtensionType):
  # [CLASS DOC]
  # Fixture type with a self-referential field z (given by name) and a tensor field n; the
  # partner of ForwardRefA in forward reference tests.
  z: 'ForwardRefB'
  n: tensor.Tensor


class ExtensionTypeWithTensorDefault(extension_type.ExtensionType):
  # [CLASS DOC]
  # Fixture type with tensor-annotated fields whose defaults are plain python values (an
  # integer and a list of strings); used to check defaults are not converted to tensors at
  # class creation time.
  x: tensor.Tensor = 5
  y: tensor.Tensor = ['a', 'b', 'c']


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test class for the extension type base class, run in graph and eager modes. Covers
  # attributes, immutability, constructors and signatures, validation, equality, use inside
  # compiled functions, control flow, nest utilities, specs, pickling, dispatch and
  # more. Has a nested Node fixture type used by the default value tests.

  def testAttributeAccessors(self):
    # [FUNCTION DOC]
    # Builds a masked tensor and its packed form, and asserts that for both the values and
    # mask fields are tensors holding the supplied data.
    mt1 = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    mt2 = extension_type.pack(mt1)

    for mt in [mt1, mt2]:
      self.assertIsInstance(mt.values, tensor.Tensor)
      self.assertAllEqual(mt.values, [1, 2, 3, 4])
      self.assertIsInstance(mt.mask, tensor.Tensor)
      self.assertAllEqual(mt.mask, [True, True, False, True])

  def testAttributesAreImmutable(self):
    # [FUNCTION DOC]
    # For an unpacked and a packed masked tensor, asserts that setting a new attribute,
    # replacing an existing field, and deleting a field each raise an attribute error saying
    # attributes cannot be mutated outside the custom constructor.
    mt1 = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    mt2 = extension_type.pack(mt1)

    for mt in [mt1, mt2]:
      with self.assertRaisesRegex(
          AttributeError,
          'Cannot mutate attribute `score` outside the custom constructor of'
          ' ExtensionType',
      ):
        mt.score = 12
      with self.assertRaisesRegex(
          AttributeError,
          'Cannot mutate attribute `values` outside the custom constructor of'
          ' ExtensionType',
      ):
        mt.values = constant_op.constant([4, 3, 2, 1])
      with self.assertRaisesRegex(
          AttributeError,
          'Cannot mutate attribute `values` outside the custom constructor of'
          ' ExtensionType',
      ):
        del mt.values

  def testClassAndStaticMethod(self):
    # [FUNCTION DOC]
    # Uses the alternate constructor and the static method of the richer masked tensor and
    # asserts the mask is all true and the static method returns the expected address text.
    mt = MaskedTensorV2.from_full_tensor([1, 2, 3, 4])
    self.assertAllEqual(mt.mask, [True, True, True, True])
    self.assertEqual(mt.doc_link(), 'http://example.com/masked_tensor')

  def testRepr(self):
    # [FUNCTION DOC]
    # Builds a simple masked tensor and asserts its default printed form lists the type name
    # and each field name with its value's printed form.
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensorV1(values, mask)
    expected = f'MaskedTensorV1(values={values!r}, mask={mask!r})'
    self.assertEqual(expected, repr(mt))

  def testEagerRepr(self):
    # [FUNCTION DOC]
    # Builds a richer masked tensor and asserts its printed form: masked-array style text in
    # eager mode, default field listing in graph mode. The same assertion is repeated twice
    # (the second is redundant, perhaps to check repeated printing is stable).
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensorV2(values, mask)
    if context.executing_eagerly():
      expected = '<MaskedTensorV2 [1, 2, _, 4]>'
    else:
      expected = f'MaskedTensorV2(values={values!r}, mask={mask!r})'

    self.assertEqual(expected, repr(mt))
    self.assertEqual(expected, repr(mt))

  def testAsDict(self):
    # [FUNCTION DOC]
    # Converts a masked tensor to a mapping of field names to values with the as-dictionary
    # utility and asserts it contains exactly the values and mask entries.
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensorV1(values, mask)
    mt_dict = extension_type.as_dict(mt)
    self.assertEqual({'values': values, 'mask': mask}, mt_dict)

  def testConstructorSignature(self):
    # [FUNCTION DOC]
    # Defines a local type with two tensor fields and a defaulted tuple field (given as a
    # list), then asserts the generated constructor signature has positional-or-keyword
    # parameters with matching annotations, the default converted to a tuple, and the
    # type itself as the return annotation.
    class MyType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type: two tensor fields and a defaulted tuple field.
      x: tensor.Tensor
      y: tensor.Tensor
      z: typing.Tuple[typing.Union[int, str], ...] = [1, 'two', 3]

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter(
            'x', POSITIONAL_OR_KEYWORD, annotation=tensor.Tensor),
        tf_inspect.Parameter(
            'y', POSITIONAL_OR_KEYWORD, annotation=tensor.Tensor),
        tf_inspect.Parameter(
            'z',
            POSITIONAL_OR_KEYWORD,
            annotation=typing.Tuple[typing.Union[int, str], ...],
            default=(1, 'two', 3),
        ),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType
    )
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.__init__))

  def testConstructorSignatureWithKeywordOnlyArgs(self):
    # [FUNCTION DOC]
    # Defines a local type with a required field, a defaulted field, then a required field,
    # and asserts the generated constructor makes the last (required after default) parameter
    # keyword-only while the others stay positional-or-keyword.
    class MyType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a required integer, a defaulted string and a required tensor.
      a: int
      b: str = 'Hello world'
      c: tensor.Tensor

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('a', POSITIONAL_OR_KEYWORD, annotation=int),
        tf_inspect.Parameter(
            'b', POSITIONAL_OR_KEYWORD, annotation=str, default='Hello world'
        ),
        tf_inspect.Parameter('c', KEYWORD_ONLY, annotation=tensor.Tensor),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType
    )
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.__init__))

  def testConstructorSignatureWithDefaultForTensorField(self):
    # [FUNCTION DOC]
    # Creates the fixture with tensor-typed fields using only defaults. Asserts the signature
    # still carries the raw python defaults (integer and list, not tensors), and that the
    # constructed instance's field works in arithmetic, proving it was converted at
    # construction time rather than at class creation time.
    a = ExtensionTypeWithTensorDefault()

    # Check that the default values were *not* converted to Tensors:
    sig = tf_inspect.signature(ExtensionTypeWithTensorDefault.__init__)
    self.assertIsInstance(sig.parameters['x'].default, int)
    self.assertIsInstance(sig.parameters['y'].default, list)

    # The following would fail with "RuntimeError: Attempting to capture an
    # EagerTensor without building a function" if we converted the default
    # value to a Tensor when we built the type.
    self.assertAllEqual(a.x + constant_op.constant(3), 8)

  def testConstructorSignatureWithAnnotatedTensorField(self):
    # [FUNCTION DOC]
    # Defines a local type whose annotations carry extra metadata wrappers (including inside an
    # optional), and asserts the generated constructor signature strips the metadata, leaving
    # plain annotations and the given defaults.
    class MyType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with metadata-annotated tensor, string and optional integer fields.
      a: typing_extensions.Annotated[tensor.Tensor, 'metadata']
      b: typing_extensions.Annotated[str, 'metadata'] = 'Hello world'
      c: typing.Optional[typing_extensions.Annotated[int, 'metadata']] = None

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter(
            'a', POSITIONAL_OR_KEYWORD, annotation=tensor.Tensor),
        tf_inspect.Parameter(
            'b', POSITIONAL_OR_KEYWORD, annotation=str, default='Hello world'
        ),
        tf_inspect.Parameter(
            'c',
            POSITIONAL_OR_KEYWORD,
            annotation=typing.Optional[int],
            default=None,
        ),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType
    )
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.__init__))

  def testEmptyType(self):
    # [FUNCTION DOC]
    # Defines a type with no fields, asserts it reports no fields, can be instantiated, and
    # prints as its qualified local name with empty parentheses.
    class EmptyType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with no fields.
      pass

    self.assertEmpty(EmptyType._tf_extension_type_fields())
    x = EmptyType()
    self.assertEqual(
        repr(x), 'ExtensionTypeTest.testEmptyType.<locals>.EmptyType()'
    )

  def testCustomConstrutor(self):
    # [FUNCTION DOC]
    # Defines a type with a custom constructor taking only a tensor-like input and computing
    # the mean and maximum fields from it, builds an instance from a nested list and asserts
    # the three fields hold the data, 3.5 and 6.
    class SummarizedTensor(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with values, mean and max tensor fields, derived by its constructor.
      values: tensor.Tensor
      mean: tensor.Tensor
      max: tensor.Tensor

      def __init__(self, values):
        # [FUNCTION DOC]
        # Custom constructor. Input: values, tensor-like. Stores the converted values tensor
        # and its mean and maximum reductions as fields. Returns nothing.
        self.values = ops.convert_to_tensor(values)
        self.mean = math_ops.reduce_mean(values)
        self.max = math_ops.reduce_max(values)

    x = SummarizedTensor([[1.0, 2, 3], [4, 5, 6]])
    self.assertAllEqual(x.values, [[1.0, 2, 3], [4, 5, 6]])
    self.assertAllEqual(x.mean, 3.5)
    self.assertAllEqual(x.max, 6)

  class Node(extension_type.ExtensionType):
    # [CLASS DOC]
    # Fixture tree node type: a tensor value x, an optional string label y (default none) and a
    # tuple of child nodes (default empty), referring to itself by qualified name.
    x: tensor.Tensor
    y: typing.Optional[str] = None
    children: typing.Tuple['ExtensionTypeTest.Node', ...] = ()

  def testConstructorWithDefaultValues(self):
    # [FUNCTION DOC]
    # Builds nodes with only the required value, with a positional label, and with only a
    # keyword children tuple, asserting defaults (no label, empty children) apply exactly when
    # the arguments are omitted.
    a = ExtensionTypeTest.Node(5)
    self.assertAllEqual(a.x, 5)
    self.assertIsNone(a.y)
    self.assertEqual(a.children, ())

    b = ExtensionTypeTest.Node(6, 'blue')
    self.assertAllEqual(b.x, 6)
    self.assertEqual(b.y, 'blue')
    self.assertEqual(b.children, ())

    c = ExtensionTypeTest.Node(7, children=(a, b))
    self.assertAllEqual(c.x, 7)
    self.assertIsNone(c.y)
    self.assertEqual(c.children, (a, b))

  def testCustomConstrutorCantMutateNestedValues(self):
    # [FUNCTION DOC]
    # Defines an outer type whose custom constructor tries to modify a field of the nested
    # already-built inner value, and asserts construction raises an attribute error saying the
    # attribute cannot be mutated outside the custom constructor of the inner type.
    class Foo(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a single integer field.
      x: int

    class Bar(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type wrapping a Foo whose constructor wrongly mutates the Foo.
      foo: Foo

      def __init__(self, foo):
        # [FUNCTION DOC]
        # Custom constructor. Input: foo, an already constructed Foo. Attempts to assign to
        # its field, which is expected to raise an attribute error (the test relies on this).
        foo.x = 33  # This raises an exception

    with self.assertRaisesRegex(
        AttributeError,
        'Cannot mutate attribute `x` outside the custom constructor of'
        ' ExtensionType',
    ):
      Bar(Foo(12))

  def testCustomValidate(self):
    # [FUNCTION DOC]
    # Defines a type with a validation hook requiring two tensors to have compatible shapes.
    # Asserts equal-length inputs are accepted and stored, and a length mismatch raises a
    # value error.
    class AlignedTensors(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with tensor fields x and y and a shape compatibility validation hook.
      x: tensor.Tensor
      y: tensor.Tensor

      def __validate__(self):
        # [FUNCTION DOC]
        # Validation hook: asserts the shapes of x and y are compatible, raising a value error
        # otherwise. Returns nothing.
        self.x.shape.assert_is_compatible_with(self.y.shape)

    aligned = AlignedTensors([1, 2, 3], ['a', 'b', 'c'])
    self.assertAllEqual(aligned.x, [1, 2, 3])
    self.assertAllEqual(aligned.y, [b'a', b'b', b'c'])

    with self.assertRaises(ValueError):
      AlignedTensors([1, 2, 3], ['a', 'b', 'c', 'd'])

  def testEquals(self):
    # [FUNCTION DOC]
    # Defines a type with tensor, tensor and string fields. Asserts equality and inequality
    # operators give expected results for equal and differing values, that broadcast-compatible
    # but differently shaped values are not equal, and that the same holds when the tensors are
    # replaced by placeholders with unknown shapes (a different internal code path).
    class MyType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a values tensor, a score tensor and a flavor string.
      values: tensor.Tensor
      score: tensor.Tensor
      flavor: str

    x1 = MyType([1, 2], 8, 'blue')
    x2 = MyType([1, 2], 8, 'blue')
    y = MyType([1, 2], 8, 'red')
    z = MyType([1, 2], 7, 'blue')
    self.assertAllEqual(x1 == x2, True)
    self.assertAllEqual(x1 != x2, False)
    self.assertAllEqual(x1 == y, False)
    self.assertAllEqual(x1 != y, True)
    self.assertAllEqual(x1 == z, False)
    self.assertAllEqual(y == z, False)

    # These are not equal, even though their values are broadcast-compatible
    # and elements are all equal when we broadcast.  Shapes must match.
    a = MyType([1, 1, 1, 1], 0, 'x')
    b = MyType([[1, 1, 1, 1]], 0, 'x')
    c = MyType([[1, 1], [1, 1]], 0, 'x')
    self.assertAllEqual(a == b, False)
    self.assertAllEqual(a == c, False)
    self.assertAllEqual(b == c, False)

    # Test with unknown shapes (executes a different codepath).
    a_ph = replace_tensors_with_placeholders(a)
    b_ph = replace_tensors_with_placeholders(b)
    c_ph = replace_tensors_with_placeholders(c)
    self.assertAllEqual(a_ph == b_ph, False)
    self.assertAllEqual(a_ph == c_ph, False)
    self.assertAllEqual(b_ph == c_ph, False)

  def testPassIntoTfFunction(self):
    # [FUNCTION DOC]
    # Passes a masked tensor, unpacked and packed, as argument to a compiled function that
    # fills masked entries with 99, and asserts both calls return the expected values.
    @def_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Compiled function. Input: a masked tensor. Returns it with masked-out entries replaced
      # by 99.
      return x.with_default(99)

    mt = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    self.assertAllEqual([1, 2, 99, 4], fn(mt))
    self.assertAllEqual([1, 2, 99, 4], fn(extension_type.pack(mt)))

  def testReturnFromTfFunction(self):
    # [FUNCTION DOC]
    # Calls two compiled functions that return a masked tensor (one plain, one packed) built
    # from an input tensor, and asserts the results are masked tensor instances with the
    # expected values and mask, and that the second result is packed.
    @def_function.function
    def mask_neg_values(x):
      # [FUNCTION DOC]
      # Compiled function. Input: a tensor. Returns a masked tensor hiding non-positive entries.
      return MaskedTensorV2(x, x > 0)

    @def_function.function
    def mask_neg_values_packed(x):
      # [FUNCTION DOC]
      # Compiled function. Input: a tensor. Returns the packed form of a masked tensor hiding
      # non-positive entries.
      return extension_type.pack(MaskedTensorV2(x, x > 0))

    expected = MaskedTensorV2([5, 8, -3, 9], [True, True, False, True])

    actual1 = mask_neg_values(constant_op.constant([5, 8, -3, 9]))
    self.assertIsInstance(actual1, MaskedTensorV2)
    self.assertAllEqual(expected.values, actual1.values)
    self.assertAllEqual(expected.mask, actual1.mask)

    actual2 = mask_neg_values_packed(constant_op.constant([5, 8, -3, 9]))
    self.assertIsInstance(actual2, MaskedTensorV2)
    self.assertTrue(extension_type.is_packed(actual2))
    self.assertAllEqual(expected.values, actual2.values)
    self.assertAllEqual(expected.mask, actual2.mask)

  def testCaptureByTfFunction(self):
    # [FUNCTION DOC]
    # Creates an outer masked tensor captured by a compiled function that adds its values and
    # intersects its mask with the argument's, calls it, and asserts the result type, values
    # and mask match the expected broadcast result.
    x = MaskedTensorV2(
        values=[[1, 2, 3], [4, 5, 6]],
        mask=[[True, True, True], [True, False, True]],
    )

    @def_function.function
    def add_to_x(y):
      # [FUNCTION DOC]
      # Compiled function capturing the outer value. Input: a masked tensor y. Returns a new
      # masked tensor with summed values and a logical-and of the masks.
      return MaskedTensorV2(x.values + y.values, x.mask & y.mask)

    actual = add_to_x(MaskedTensorV2([10, 20, 30], [False, True, True]))
    expected = MaskedTensorV2(
        values=[[11, 22, 33], [14, 25, 36]],
        mask=[[False, True, True], [False, False, True]],
    )
    self.assertIsInstance(actual, MaskedTensorV2)
    self.assertAllEqual(expected.values, actual.values)
    self.assertAllEqual(expected.mask, actual.mask)

  def testTfFunctionArgMutationError(self):
    # [FUNCTION DOC]
    # Calls a compiled function that appends to its list argument and asserts a value error
    # saying the function should not modify its arguments.
    @def_function.function
    def fn_with_side_effect(mts):
      # [FUNCTION DOC]
      # Compiled function that deliberately mutates its input list by appending a new masked
      # tensor (not allowed). Returns nothing.
      mts.append(MaskedTensorV1(mts[0].values * 2, mts[0].mask))

    with self.assertRaisesRegex(ValueError, 'should not modify'):
      fn_with_side_effect([MaskedTensorV1([10, 20, 30], [False, True, True])])

  def testNestPackUnpack(self):
    # [FUNCTION DOC]
    # Flattens a store value (a name tensor and a mapping of price tensors) into its component
    # tensors, rebuilds it from both the value and its type specification, and asserts the
    # component count and order (mapping sorted by key) and that all fields survive.
    class CandyStore(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a name tensor and a mapping from strings to price tensors.
      name: tensor.Tensor
      prices: typing.Mapping[str, tensor.Tensor]

    store = CandyStore('Yum', {'gum': [0.42, 0.48], 'chocolate': [0.83, 1.02]})
    components = nest.flatten(store, expand_composites=True)
    repacked_1 = nest.pack_sequence_as(
        store, components, expand_composites=True
    )
    repacked_2 = nest.pack_sequence_as(
        store._type_spec, components, expand_composites=True
    )

    # Note: dicts get sorted by key.
    self.assertLen(components, 3)
    self.assertAllEqual(components[0], b'Yum')
    self.assertAllClose(components[1], [0.83, 1.02])
    self.assertAllClose(components[2], [0.42, 0.48])

    for repacked in [repacked_1, repacked_2]:
      self.assertAllEqual(repacked.name, b'Yum')
      self.assertAllClose(repacked.prices['gum'], [0.42, 0.48])
      self.assertAllClose(repacked.prices['chocolate'], [0.83, 1.02])

  def testSimpleCond(self):
    # [FUNCTION DOC]
    # Uses a conditional op with constant predicates choosing between two simple masked
    # tensors, and asserts the chosen values and masks match the selected branch.
    x = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])
    y = MaskedTensorV1([5, 6, 7, 8], [False, True, True, False])

    x_2 = tf_cond.cond(constant_op.constant(True), lambda: x, lambda: y)
    y_2 = tf_cond.cond(constant_op.constant(False), lambda: x, lambda: y)

    self.assertAllEqual(x.values, x_2.values)
    self.assertAllEqual(x.mask, x_2.mask)
    self.assertAllEqual(y.values, y_2.values)
    self.assertAllEqual(y.mask, y_2.mask)

  def testComplexCond(self):
    # [FUNCTION DOC]
    # Uses a conditional op whose branches build new masked tensors from a captured one.
    # Asserts (see following lines) the result of each branch.
    mt = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])

    def true_fn():
      # [FUNCTION DOC]
      # Branch function with no inputs. Returns a masked tensor with masked-out values replaced
      # by minus one and a mask marking values greater than three.
      return MaskedTensorV1(
          array_ops.where_v2(mt.mask, mt.values, -1), mt.values > 3
      )

    def false_fn():
      # [FUNCTION DOC]
      # Branch function with no inputs. Returns a masked tensor with masked-in values replaced
      # by 100 and others doubled, with a mask derived from the captured value.
      return MaskedTensorV1(
          array_ops.where_v2(mt.mask, 100, mt.values * 2),
          math_ops.logical_not(mt.mask),
      )

    x = tf_cond.cond(constant_op.constant(True), true_fn, false_fn)
    y = tf_cond.cond(constant_op.constant(False), true_fn, false_fn)

    self.assertAllEqual(x.values, [1, -1, 3, -1])
    self.assertAllEqual(x.mask, [False, False, False, True])
    self.assertAllEqual(y.values, [100, 4, 100, 8])
    self.assertAllEqual(y.mask, [False, True, False, True])

  def testCondAutograph(self):
    # [FUNCTION DOC]
    # Calls a compiled function that branches on a tensor condition (converted to graph
    # conditionals by autograph) and returns different masked tensors per branch. Asserts the
    # values and mask of the taken branch.
    @def_function.function
    def fn(mt):
      # [FUNCTION DOC]
      # Compiled function. Input: a masked tensor. If its last value exceeds three returns a
      # masked tensor with masked-out values set to minus one, else one with doubled values.
      if mt.values[3] > 3:
        return MaskedTensorV1(
            array_ops.where_v2(mt.mask, mt.values, -1), mt.values > 3
        )
      else:
        return MaskedTensorV1(
            array_ops.where_v2(mt.mask, 100, mt.values * 2), not mt.mask
        )

    x = fn(MaskedTensorV1([1, 2, 3, 4], [True, False, True, False]))
    self.assertAllEqual(x.values, [1, -1, 3, -1])
    self.assertAllEqual(x.mask, [False, False, False, True])

  def testCondTypeMismatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the eager-mode guard tests the method object itself rather than calling it,
    # so it is always true and the test returns immediately in every mode; the assertions below
    # never run.
    # Intended scenario: conditionals whose branches return mismatched structures (different
    # extension types, or an extension type versus a plain tensor) should raise a value or
    # type error mentioning incompatible return values or types of the two branches.
    if context.executing_eagerly:
      # In eager mode, tf.cond eagerly runs either true_fn or false_fn, and
      # ignores the other one; so it doesn't detect any type mismatches
      # between the two outcomes.  (See _eager_cond_implementation in
      # cond.py.)
      return

    a = lambda: MaskedTensorV1([1, 2, 3], [True, True, False])
    b = lambda: MaskedTensorV1(['a', 'b', 'c'], [False, True, True])
    c = lambda: MaskedTensorV2([4, 5, 6], [True, True, False])
    d = lambda: constant_op.constant([7, 8, 9])

    with self.assertRaisesRegex(
        ValueError,
        'Incompatible return values of true_fn and false_fn: The two '
        "structures don't have the same nested structure",
    ):
      tf_cond.cond(constant_op.constant(True), a, b)
    with self.assertRaisesRegex(
        TypeError,
        'Incompatible return types of true_fn and false_fn: The two '
        "structures don't have the same nested structure",
    ):
      tf_cond.cond(constant_op.constant(True), a, c)
    with self.assertRaisesRegex(
        ValueError,
        'Incompatible return values of true_fn and false_fn: The two '
        "structures don't have the same nested structure",
    ):
      tf_cond.cond(constant_op.constant(True), a, d)

  def testCondPacked(self):
    # [FUNCTION DOC]
    # Uses conditionals with packed masked tensors as branch results and as captured values.
    # Asserts the selected branch's values and mask, that sizes of fields of an unpacked and a
    # packed value can be returned from branches, and (per the inline note) that attribute
    # access on a packed value must not cache the unpacked result across graph scopes.
    x = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    y = MaskedTensorV2([5, 6, 7, 8], [False, True, True, False])
    x = extension_type.pack(x)
    y = extension_type.pack(y)

    x_2 = tf_cond.cond(constant_op.constant(True), lambda: x, lambda: y)
    y_2 = tf_cond.cond(constant_op.constant(False), lambda: x, lambda: y)

    self.assertAllEqual(x.values, x_2.values)
    self.assertAllEqual(x.mask, x_2.mask)
    self.assertAllEqual(y.values, y_2.values)
    self.assertAllEqual(y.mask, y_2.mask)

    a = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    b = extension_type.pack(a)
    b = tf_cond.cond(
        constant_op.constant(True),
        lambda: array_ops.size(a.mask),
        lambda: array_ops.size(a.values),
    )
    self.assertAllEqual(b, 4)

    # Note: the following example would fail (with `Retval[0] does not have a
    # value`) if `ExtensionType.__getattr__` cached the results of unpacking
    # the value.  See the comment in `ExtensionType.__getattr__` for details.
    c = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    c = extension_type.pack(c)
    d = tf_cond.cond(
        constant_op.constant(False),
        lambda: array_ops.size(c.mask),
        lambda: array_ops.size(c.values),
    )
    self.assertAllEqual(d, 4)

  def testWhileLoop(self):
    # [FUNCTION DOC]
    # Runs a while loop for ten iterations carrying a counter and a masked tensor whose values
    # double each step. Asserts the result is a masked tensor with values scaled by 1024 and
    # the mask unchanged.
    x = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])

    cond = lambda i, x: i < 10
    body = lambda i, x: (i + 1, MaskedTensorV1(x.values * 2, x.mask))
    _, y = while_loop.while_loop_v2(cond, body, [0, x])

    self.assertIsInstance(y, MaskedTensorV1)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testWhileLoopAutograph(self):
    # [FUNCTION DOC]
    # Calls a compiled function whose python loop over a tensor range (converted by autograph)
    # doubles a masked tensor's values each iteration; asserts values scale by 1024 for ten
    # iterations, the mask is unchanged and the result type is preserved.
    @def_function.function
    def fn(x, n):
      # [FUNCTION DOC]
      # Compiled function. Inputs: a masked tensor x and an iteration count n. Returns x after
      # n rounds of doubling its values.
      for _ in math_ops.range(n):
        x = MaskedTensorV1(x.values * 2, x.mask)
      return x

    y = fn(MaskedTensorV1([1, 2, 3, 4], [True, False, True, False]), 10)
    self.assertIsInstance(y, MaskedTensorV1)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testWhileLoopTypeMismatch(self):
    # [FUNCTION DOC]
    # Runs a while loop whose body changes the loop variable's structure (masked tensor in,
    # plain tensor out). Asserts a value error stating the structures do not match.
    x = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])

    cond = lambda i, x: i < 10

    def body(i, x):
      # [FUNCTION DOC]
      # KNOWN ISSUE: returns a single value rather than the (counter, value) pair the loop
      # expects, which is part of what the test relies on to trigger the mismatch error.
      # Loop body. Inputs: counter i and loop variable x. Returns a plain tensor when x is a
      # masked tensor, otherwise a masked tensor, so the structure differs from the input.
      if isinstance(x, MaskedTensorV1):
        return x.values * 2
      else:
        return MaskedTensorV1(x, x > i)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure"
    ):
      while_loop.while_loop_v2(cond, body, [0, x])

  def testWhileLoopPacked(self):
    # [FUNCTION DOC]
    # Runs a ten step while loop with a packed masked tensor as loop variable, doubling values
    # and repacking each step. Asserts the result is a masked tensor with values scaled by 1024
    # and the mask unchanged.
    x = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    x = extension_type.pack(x)
    cond = lambda i, x: i < 10

    def body(i, x):
      # [FUNCTION DOC]
      # Loop body. Inputs: counter i and packed masked tensor x. Returns the incremented
      # counter and a packed masked tensor with doubled values and the same mask.
      return i + 1, extension_type.pack(MaskedTensorV2(x.values * 2, x.mask))

    _, y = while_loop.while_loop_v2(cond, body, [0, x])
    self.assertIsInstance(y, MaskedTensorV2)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testNestedFields(self):
    # [FUNCTION DOC]
    # Builds a type with deeply nested annotations (a tuple of tuples holding a string, a tensor
    # and a mapping to dense-or-ragged tensors, plus a mapping to tensors) from plain python
    # data. Asserts values are converted to tensors or ragged tensors at the right places,
    # mappings become immutable dictionaries, and the printed form matches an expected pattern.
    PossiblyRaggedTensor = typing.Union[
        tensor.Tensor, ragged_tensor.RaggedTensor]
    ToyFeatures = typing.Mapping[str, PossiblyRaggedTensor]

    class ToyInfo(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type: version string, nested tuple of toy descriptions, and a mapping from
      # box names to tensors.
      version: str
      toys: typing.Tuple[typing.Tuple[str, tensor.Tensor, ToyFeatures], ...]
      boxes: typing.Mapping[str, tensor.Tensor]

    authors = [[b'A', b'Aardvark'], [b'Z', b'Zhook']]
    toys = [
        ('car', 1.0, {'size': [8, 3, 2], 'color': [0.3, 0.2, 0.8]}),
        ('book', 3.7, {'authors': ragged_factory_ops.constant(authors)}),
    ]
    boxes = {'green': ['car'], 'blue': ['car', 'book', 'book']}
    toy_info = ToyInfo(version='1.0 alpha', toys=toys, boxes=boxes)

    self.assertEqual(toy_info.version, '1.0 alpha')
    self.assertEqual(toy_info.toys[0][0], 'car')
    self.assertIsInstance(toy_info.toys[0][1], tensor.Tensor)
    self.assertAllEqual(toy_info.toys[0][1], 1.0)
    self.assertEqual(set(toy_info.toys[0][2].keys()), {'size', 'color'})
    self.assertIsInstance(toy_info.toys[0][2]['size'], tensor.Tensor)
    self.assertAllEqual(toy_info.toys[0][2]['size'], [8, 3, 2])
    self.assertIsInstance(
        toy_info.toys[1][2]['authors'], ragged_tensor.RaggedTensor
    )
    self.assertAllEqual(toy_info.toys[1][2]['authors'], authors)
    self.assertAllEqual(toy_info.boxes['green'], [b'car'])
    self.assertAllEqual(toy_info.boxes['blue'], ['car', 'book', 'book'])

    expected_repr = (
        r"ToyInfo\(version='1.0 alpha', toys=\("
        r"\('car', <tf.Tensor[^>]*>, ImmutableDict\("
        r"{'size': <tf.Tensor[^>]*>, 'color': <tf.Tensor[^>]*>}\)\), "
        r"\('book', <tf.Tensor[^>]*>, ImmutableDict\("
        r"{'authors': (<tf.RaggedTensor[^>]*>|tf.RaggedTensor\(.*\))}\)\)\), "
        r'boxes=ImmutableDict\('
        r"{'green': <tf.Tensor[^>]*>, 'blue': <tf.Tensor[^>]*>}\)\)"
    )

    self.assertRegex(repr(toy_info), expected_repr)

  def testNestedExtensionTypes(self):
    # [FUNCTION DOC]
    # Builds extension types that contain other extension types inside tuples and mappings,
    # constructs an instance from nested values, passes it to a compiled function that sums the
    # toy prices, and asserts the sum is 4.7.
    PossiblyMaskedTensor = typing.Union[tensor.Tensor, MaskedTensorV1]

    class Toy(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type: name string, price tensor and mapping of feature tensors.
      name: str
      price: tensor.Tensor
      features: typing.Mapping[str, PossiblyMaskedTensor]

    class Box(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type holding a contents tensor.
      contents: tensor.Tensor

    class ToyInfo(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type: version string, tuple of toys and mapping from names to boxes.
      version: str
      toys: typing.Tuple[Toy, ...]
      boxes: typing.Mapping[str, Box]

    authors = MaskedTensorV1(
        values=[[b'A', b'Quincy', b'Aardvark'], [b'Z', b'Zhook', b'']],
        mask=[[True, True, True], [True, True, False]],
    )
    toys = [
        Toy('car', 1.0, {'size': [8, 3, 2], 'color': [0.3, 0.2, 0.8]}),
        Toy(name='book', price=3.7, features={'authors': authors}),
    ]
    boxes = {
        'green': Box(['car']),
        'blue': Box(contents=['car', 'book', 'book']),
    }
    toy_info = ToyInfo(version='1.0 alpha', toys=toys, boxes=boxes)

    @def_function.function
    def fn(info):
      # [FUNCTION DOC]
      # Compiled function. Input: a toy info value. Returns the sum of all toy prices.
      prices = [toy.price for toy in info.toys]
      return math_ops.reduce_sum(array_ops_stack.stack(prices))

    self.assertAllClose(fn(toy_info), 4.7)

  def testNestedCustomConstructor(self):
    # [FUNCTION DOC]
    # Defines a toy type and a box type that both have custom constructors; the box builds its
    # toys applying per-name discounts. Asserts four toys exist and discounted ones have the
    # discount suffix in their names.
    class Toy(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a name string and price tensor and a discounting constructor.
      name: str
      price: tensor.Tensor

      def __init__(self, name, price, discount=0):
        # [FUNCTION DOC]
        # Custom constructor. Inputs: name, price, and optional discount fraction (default
        # zero). When the discount is nonzero appends a discount suffix to the name and scales
        # the price down. Stores name and price. Returns nothing.
        if discount:
          name += ' (discounted)'
          price *= 1 - discount
        self.name = name
        self.price = price

    class ToyBox(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type holding a tuple of toys, built by its custom constructor.
      toys: typing.Tuple[Toy, ...]

      def __init__(self, name_to_price, name_to_discount):
        # [FUNCTION DOC]
        # Custom constructor. Inputs: mapping from toy names to prices, and mapping from names
        # to discount fractions (missing names mean no discount). Builds one toy per entry and
        # stores the list as the toys field. Returns nothing.
        self.toys = [
            Toy(name, price, name_to_discount.get(name, 0))
            for (name, price) in name_to_price.items()
        ]

    toy_box = ToyBox(
        {'car': 8.3, 'truck': 5.9, 'puzzle': 5.3, 'jacks': 2.8},
        {'puzzle': 0.2, 'truck': 0.3},
    )
    self.assertLen(toy_box.toys, 4)
    self.assertEqual(
        set(toy.name for toy in toy_box.toys),
        {'car', 'truck (discounted)', 'puzzle (discounted)', 'jacks'},
    )

  def testExtensionTypeWithMathOperators(self):
    # [FUNCTION DOC]
    # Temporarily registers an addition dispatcher for the richer masked tensor type, adds two
    # such values with the operator, and asserts the values are summed and the masks combined
    # with logical and. The dispatcher is removed on exit.
    def masked_add(x, y, name=None):
      # [FUNCTION DOC]
      # KNOWN ISSUE: the support check is written as "x is not masked and y is masked", which
      # looks like it was meant to reject when either operand is not a masked tensor; as
      # written it only rejects the case where only y is masked.
      # Dispatcher for addition. Inputs: x, y operands and an ignored name. Returns the
      # not-supported marker in that case, otherwise a masked tensor with summed values and
      # the logical and of the masks.
      del name
      if not isinstance(x, MaskedTensorV2) and isinstance(y, MaskedTensorV2):
        return dispatch.OpDispatcher.NOT_SUPPORTED
      return MaskedTensorV2(x.values + y.values, x.mask & y.mask)

    with temporarily_add_dispatch(math_ops.add, MaskedTensorV2, masked_add):
      x = MaskedTensorV2([[1, 2], [3, 4]], [[True, False], [True, True]])
      y = MaskedTensorV2([[3, 4], [5, 6]], [[True, True], [False, True]])
      z = x + y
      self.assertAllEqual(z.values, [[4, 6], [8, 10]])
      self.assertAllEqual(z.mask, [[True, False], [False, True]])

  def testGetExtensionTypeFields(self):
    # [FUNCTION DOC]
    # Asks for the field list on both the type and an instance of the simple masked tensor and
    # asserts there are two fields, values and mask, each tensor-typed with no default.
    # Can be called on a type or an instance:
    fields_1 = MaskedTensorV1._tf_extension_type_fields()
    fields_2 = MaskedTensorV1([0], [True])._tf_extension_type_fields()

    for fields in [fields_1, fields_2]:
      self.assertLen(fields, 2)
      self.assertEqual(fields[0].name, 'values')
      self.assertEqual(fields[0].value_type, tensor.Tensor)
      self.assertEqual(fields[0].default, fields[0].NO_DEFAULT)
      self.assertEqual(fields[1].name, 'mask')
      self.assertEqual(fields[1].value_type, tensor.Tensor)
      self.assertEqual(fields[1].default, fields[0].NO_DEFAULT)

  def testHasExtensionTypeField(self):
    # [FUNCTION DOC]
    # Asserts the has-field query returns true for the two declared names and false for an
    # unknown name, on both the type and an instance.
    self.assertTrue(MaskedTensorV1._tf_extension_type_has_field('values'))
    self.assertTrue(MaskedTensorV1._tf_extension_type_has_field('mask'))
    self.assertFalse(MaskedTensorV1._tf_extension_type_has_field('labels'))

    mt = MaskedTensorV1([0], [True])
    self.assertTrue(mt._tf_extension_type_has_field('values'))
    self.assertTrue(mt._tf_extension_type_has_field('mask'))
    self.assertFalse(mt._tf_extension_type_has_field('labels'))

  def testForwardReferences(self):
    # [FUNCTION DOC]
    # Asserts that the two mutually referencing fixture types have their by-name annotations
    # resolved to the real classes in their field lists, while the generated constructor
    # signature keeps the unresolved names as string annotations.
    A, B = ForwardRefA, ForwardRefB

    self.assertEqual(
        A._tf_extension_type_fields(),
        (
            extension_type_field.ExtensionTypeField(
                'x', typing.Tuple[typing.Union[A, B], ...]
            ),
            extension_type_field.ExtensionTypeField('y', B),
        ),
    )
    self.assertEqual(
        B._tf_extension_type_fields(),
        (
            extension_type_field.ExtensionTypeField('z', B),
            extension_type_field.ExtensionTypeField('n', tensor.Tensor),
        ),
    )

    # Check the signature.
    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter(
            'x',
            POSITIONAL_OR_KEYWORD,
            annotation=typing.Tuple[
                typing.Union['ForwardRefA', 'ForwardRefB'], ...
            ],
        ),
        tf_inspect.Parameter(
            'y', POSITIONAL_OR_KEYWORD, annotation='ForwardRefB'
        ),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=A
    )
    self.assertEqual(tf_inspect.signature(A.__init__), expected_sig)

  def testUnresolvedForwardReference(self):
    # [FUNCTION DOC]
    # Defines a type whose field refers by name to a nonexistent type (a deliberate typo) and
    # a real car type; asserts constructing the broken type raises a type error.
    class Broken(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a field annotated by an unresolvable name.
      x: 'Cra'  # note: intentional typo for Car.

    class Car(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a float speed field.
      speed: float

    with self.assertRaises(TypeError):
      Broken(x=Car(3.8))

  def testUnsupportedAnnotations(self):
    # [FUNCTION DOC]
    # Asserts that defining a type with a list annotation, and one with a union containing a
    # tuple of complex numbers, each raises a type error naming the field and saying the
    # annotation is unsupported.
    with self.assertRaisesRegex(
        TypeError, "In field 'values': Unsupported type annotation"
    ):

      class MyType1(extension_type.ExtensionType):  # pylint: disable=unused-variable
        # [CLASS DOC]
        # Local type whose definition is expected to fail: list-typed field.
        values: typing.List[tensor.Tensor]

    with self.assertRaisesRegex(
        TypeError, "In field 'xyz': Unsupported type annotation"
    ):

      class MyType2(extension_type.ExtensionType):  # pylint: disable=unused-variable
        # [CLASS DOC]
        # Local type whose definition is expected to fail: union with unsupported complex tuple.
        xyz: typing.Union[typing.Tuple[complex, ...], int]

  def testCantUseReservedName(self):
    # [FUNCTION DOC]
    # Asserts that defining a type with a field named like a reserved base-class attribute, a
    # field with the reserved extension prefix, or a method that shadows a base-class method
    # name each raises a value error stating the field annotations are invalid and the name
    # is reserved.
    with self.assertRaisesRegex(
        ValueError,
        'The field annotations for MyType1 are invalid. '
        "Field '_to_components' is reserved",
    ):

      class MyType1(extension_type.ExtensionType):  # pylint: disable=unused-variable
        # [CLASS DOC]
        # Local type whose definition is expected to fail: reserved field name.
        _to_components: int

    with self.assertRaisesRegex(
        ValueError,
        'The field annotations for MyType2 are invalid. '
        "Field '_tf_extension_type_foo' is reserved",
    ):

      class MyType2(extension_type.ExtensionType):  # pylint: disable=unused-variable
        # [CLASS DOC]
        # Local type whose definition is expected to fail: field with the reserved prefix.
        _tf_extension_type_foo: int

    with self.assertRaisesRegex(
        ValueError,
        'The field annotations for MyType3 are invalid. '
        "Field 'is_compatible_with' is reserved",
    ):

      class MyType3(extension_type.ExtensionType):  # pylint: disable=unused-variable
        # [CLASS DOC]
        # Local type whose definition is expected to fail: defines a method with a reserved name.

        def is_compatible_with(self, other):
          # [FUNCTION DOC]
          # Method that shadows a reserved base-class name; input other is ignored; always
          # returns false. Its definition is what triggers the expected error.
          return False

  def testExtensionTypeBaseClassHasNoSpec(self):
    # [FUNCTION DOC]
    # Asserts the abstract base class itself has no nested type specification attribute.
    self.assertFalse(hasattr(extension_type.ExtensionType, 'Spec'))

  def testExtensionTypeBaseConstructorRaisesException(self):
    # [FUNCTION DOC]
    # Asserts that instantiating the abstract base class directly raises an assertion error
    # saying it is an abstract base class.
    with self.assertRaisesRegex(
        AssertionError, 'ExtensionType is an abstract base class.'
    ):
      extension_type.ExtensionType()

  class ExtensionTypeWithName(extension_type.ExtensionType):
    # [CLASS DOC]
    # Fixture type with a registered name (needed for saved-model support), a field x holding a
    # tuple of a tensor and an integer, and a tensor field y.
    __name__ = 'tf.__test__.ExtensionTypeWithName'  # For SavedModel
    x: typing.Tuple[tensor.Tensor, int]
    y: tensor.Tensor

  def testSavedModelSupport(self):
    # [FUNCTION DOC]
    # Builds a module with a compiled method taking the named fixture type, traces it for two
    # argument variants, saves it to a temporary directory and (see following lines) reloads
    # it to check the extension type survives the round trip.
    class TestModule(module.Module):
      # [CLASS DOC]
      # Local module with one compiled method that adds the pieces of a named fixture value.

      @def_function.function
      def f(self, s):
        # [FUNCTION DOC]
        # Compiled method. Input: s, a named fixture value. Returns the sum of its tuple
        # elements and its tensor field.
        return s.x[0] + s.x[1] + s.y

    s1 = self.ExtensionTypeWithName((1, 2), 3)
    s2 = self.ExtensionTypeWithName((1.0, 2), [3.0, 4.0])

    m = TestModule()
    m.f.get_concrete_function(s1)
    m.f.get_concrete_function(s2)

    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    save.save(m, path)
    loaded = load.load(path)

    self.assertAllEqual(loaded.f(s1), 6)
    self.assertAllEqual(loaded.f(s2), [6.0, 7.0])

  def testPackedEncoding(self):
    # [FUNCTION DOC]
    # Packs a masked tensor into a single-component encoding and unpacks it back. Asserts the
    # flattened component count (two unpacked, one packed), that fields are readable from both
    # forms, that structure comparison fails between unpacked and packed forms but passes after
    # unpacking, and that packing a type lacking a registered name raises a value error.
    mt1 = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    self.assertLen(nest.flatten(mt1, expand_composites=True), 2)

    mt2 = extension_type.pack(mt1)
    self.assertLen(nest.flatten(mt2, expand_composites=True), 1)
    self.assertIsInstance(mt2.values, tensor.Tensor)
    self.assertAllEqual(mt2.values, [1, 2, 3, 4])
    self.assertIsInstance(mt2.mask, tensor.Tensor)
    self.assertAllEqual(mt2.mask, [True, True, False, True])

    mt3 = extension_type.unpack(mt2)
    self.assertLen(nest.flatten(mt3, expand_composites=True), 2)
    self.assertIsInstance(mt3.values, tensor.Tensor)
    self.assertAllEqual(mt3.values, [1, 2, 3, 4])
    self.assertIsInstance(mt3.mask, tensor.Tensor)
    self.assertAllEqual(mt3.mask, [True, True, False, True])

    nest.assert_same_structure(mt1, mt3, expand_composites=True)
    with self.assertRaisesRegex(ValueError, "don't have the same"):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_same_structure(mt1, mt2, expand_composites=True)

    mt4 = MaskedTensorV1([1, 2, 3, 4], [True, True, False, True])
    with self.assertRaisesRegex(
        ValueError,
        'ExtensionTypes must have a __name__ field in order to be packed.',
    ):
      extension_type.pack(mt4)

  def testSubclassing(self):
    # [FUNCTION DOC]
    # Defines a three-level hierarchy of extension types that adds fields, overrides defaults and
    # overrides a field's type. Asserts the generated constructor signatures (parameter order,
    # keyword-only status, annotations, defaults) for the middle and last classes, then builds
    # the last class with some arguments and checks every field value including defaults.
    class Instrument(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local base fixture type: name and weight tensors and a required boolean needs_case.
      name: tensor.Tensor
      weight: tensor.Tensor
      needs_case: bool

    class StringInstrument(Instrument):
      # [CLASS DOC]
      # Local subtype adding a required integer field and a default for needs_case.
      num_strings: int  # Add a new field
      needs_case: bool = True  # Override default value.

    class Violin(StringInstrument):
      # [CLASS DOC]
      # Local subtype adding a maker tensor, defaulting the string count and overriding the name
      # field with a string type and default.
      maker: tensor.Tensor
      num_strings: int = 4  # Override default value.
      name: str = 'violin'  # Override field type and default value.

    self.assertEqual(
        list(
            tf_inspect.signature(StringInstrument.__init__).parameters.values()
        ),
        [
            tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
            tf_inspect.Parameter(
                'name', POSITIONAL_OR_KEYWORD, annotation=tensor.Tensor
            ),
            tf_inspect.Parameter(
                'weight', POSITIONAL_OR_KEYWORD, annotation=tensor.Tensor
            ),
            tf_inspect.Parameter(
                'needs_case',
                POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=True,
            ),
            tf_inspect.Parameter('num_strings', KEYWORD_ONLY, annotation=int),
        ],
    )
    self.assertEqual(
        list(tf_inspect.signature(Violin.__init__).parameters.values()),
        [
            tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
            tf_inspect.Parameter(
                'name', POSITIONAL_OR_KEYWORD, annotation=str, default='violin'
            ),
            tf_inspect.Parameter(
                'weight', KEYWORD_ONLY, annotation=tensor.Tensor),
            tf_inspect.Parameter(
                'needs_case', KEYWORD_ONLY, annotation=bool, default=True
            ),
            tf_inspect.Parameter(
                'num_strings', KEYWORD_ONLY, annotation=int, default=4
            ),
            tf_inspect.Parameter(
                'maker', KEYWORD_ONLY, annotation=tensor.Tensor),
        ],
    )

    violin = Violin(weight=28, maker='Amati')
    self.assertAllEqual(violin.name, 'violin')
    self.assertAllEqual(violin.weight, 28)
    self.assertAllEqual(violin.needs_case, True)
    self.assertAllEqual(violin.num_strings, 4)
    self.assertAllEqual(violin.maker, 'Amati')


# integration test to test compatibility with high level api like Dataset
# and Keras
class ExtensionTypeIntegrationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Integration tests checking that batchable extension types work with datasets (single
  # element, batching, unbatching, ragged contents) and distributed datasets. All tests run
  # only in the second-generation (eager) behavior mode.

  @test_util.run_v2_only
  def testDataset(self):
    # [FUNCTION DOC]
    # Creates a dataset from one masked tensor element and asserts the first element read
    # back equals the original.
    mt = MaskedTensorV3([[1], [2], [3]], [[True], [False], [True]])
    ds = dataset_ops.DatasetV2.from_tensors(mt)
    self.assertEqual(next(iter(ds)), mt)

  @test_util.run_v2_only
  def testDatasetBatch(self):
    # [FUNCTION DOC]
    # Builds a dataset from one batched masked tensor and asserts that reading back, unbatching
    # (yielding the first row), slicing into rows and re-batching all give the expected masked
    # tensors.
    xs = MaskedTensorV3([[1], [2], [3]], [[True], [False], [True]])
    x0 = MaskedTensorV3(xs.values[0], xs.mask[0])

    ds = dataset_ops.DatasetV2.from_tensors(xs)
    self.assertEqual(next(iter(ds)), xs)
    ds = ds.unbatch()
    self.assertEqual(next(iter(ds)), x0)

    ds = dataset_ops.DatasetV2.from_tensor_slices(xs)
    self.assertEqual(next(iter(ds)), x0)
    ds = ds.batch(3, drop_remainder=True)
    self.assertEqual(next(iter(ds)), xs)

  @test_util.run_v2_only
  def testDatasetBatchRagged(self):
    # [FUNCTION DOC]
    # Same scenario as the dense batch test but with ragged values and mask: element
    # round trip, unbatch, from slices and re-batch are asserted to give equal masked tensors.
    xs = MaskedTensorV3(
        ragged_factory_ops.constant([[1], [2, 3], [4]]),
        ragged_factory_ops.constant([[True], [False], [True]]),
    )
    x0 = MaskedTensorV3(xs.values[0], xs.mask[0])

    ds = dataset_ops.DatasetV2.from_tensors(xs)
    self.assertEqual(next(iter(ds)), xs)
    ds = ds.unbatch()
    self.assertEqual(next(iter(ds)), x0)

    ds = dataset_ops.DatasetV2.from_tensor_slices(xs)
    self.assertEqual(next(iter(ds)), x0)
    ds = ds.batch(3, drop_remainder=True)
    self.assertEqual(next(iter(ds)), xs)

  @test_util.run_v2_only
  def testDistributedDataset(self):
    # [FUNCTION DOC]
    # Creates a mirrored strategy over two named GPU devices, distributes a batched dataset of
    # masked tensors and asserts the first per-replica result holds the expected first row's
    # values and mask. Requires two GPU devices to exist.
    strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
    mt = MaskedTensorV3([[1], [2], [3], [4]], [[True], [False], [True], [True]])
    ds = dataset_ops.DatasetV2.from_tensor_slices(mt).batch(2)
    dist_dataset = strategy.experimental_distribute_dataset(ds)
    expect = MaskedTensorV3([[1]], [[True]])
    per_replica_result = next(iter(dist_dataset))
    self.assertEqual(per_replica_result.values[0].values, expect.values[0])
    self.assertEqual(per_replica_result.values[0].mask, expect.mask[0])


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeSpecTest(
    test_util.TensorFlowTestCase, parameterized.TestCase
):
  # [CLASS DOC]
  # Tests for the automatically generated type specification classes of extension types:
  # constructor and signature, immutability, creation from a value, serialization, component
  # conversion, copying and pickling, customization and nested-spec validation. Runs in graph
  # and eager modes.

  def testSpecConstructor(self):
    # [FUNCTION DOC]
    # Builds a spec from two tensor specs, asserts the fields are stored, and that the spec of
    # a matching concrete value equals it.
    values_spec = tensor.TensorSpec([4], dtypes.float32)
    mask_spec = tensor.TensorSpec([4], dtypes.bool)
    mt_spec = MaskedTensorV1.Spec(values_spec, mask_spec)
    self.assertEqual(mt_spec.values, values_spec)
    self.assertEqual(mt_spec.mask, mask_spec)

    mt = MaskedTensorV1([1.0, 2.0, 3.0, 4.0], [True, True, False, True])
    self.assertEqual(mt._type_spec, mt_spec)

  def testSpecConstructorSignature(self):
    # [FUNCTION DOC]
    # Asserts the generated spec constructor takes all fields as plain positional-or-keyword
    # parameters (no annotations, no defaults) and returns the spec class.
    class MyType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with two tensors and a defaulted tuple field.
      x: tensor.Tensor
      y: tensor.Tensor
      z: typing.Tuple[typing.Union[int, str], ...] = [1, 'two', 3]

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('x', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('y', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('z', POSITIONAL_OR_KEYWORD),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType.Spec
    )
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.Spec.__init__))

  def testSpecAttributesAreImmutable(self):
    # [FUNCTION DOC]
    # For a spec made from a value, asserts that setting a new attribute, replacing a field or
    # deleting a field each raise an attribute error saying attributes cannot be mutated
    # outside the custom constructor of the spec class.
    mt = MaskedTensorV1([1, 2, 3, 4], [True, True, False, True])
    mt_spec = MaskedTensorV1.Spec.from_value(mt)
    with self.assertRaisesRegex(
        AttributeError,
        'Cannot mutate attribute `score` '
        'outside the custom constructor of ExtensionTypeSpec',
    ):
      mt_spec.score = 12
    with self.assertRaisesRegex(
        AttributeError,
        'Cannot mutate attribute `values` '
        'outside the custom constructor of ExtensionTypeSpec',
    ):
      mt_spec.values = constant_op.constant([4, 3, 2, 1])
    with self.assertRaisesRegex(
        AttributeError,
        'Cannot mutate attribute `values` '
        'outside the custom constructor of ExtensionTypeSpec',
    ):
      del mt_spec.values

  def testSpecFromValue(self):
    # [FUNCTION DOC]
    # Creates a spec from a concrete masked tensor and asserts its fields are tensor specs with
    # the value's shape and element types.
    mt = MaskedTensorV1([1.0, 2.0, 3.0, 4.0], [True, True, False, True])
    mt_spec = MaskedTensorV1.Spec.from_value(mt)

    expected_values_spec = tensor.TensorSpec([4], dtypes.float32)
    expected_mask_spec = tensor.TensorSpec([4], dtypes.bool)
    self.assertEqual(mt_spec.values, expected_values_spec)
    self.assertEqual(mt_spec.mask, expected_mask_spec)

  def testSpecSerialize(self):
    # [FUNCTION DOC]
    # Builds a spec with a tuple of names and a nested mapping of tensor specs, serializes it,
    # asserts the exact serialized form, deserializes and asserts equality with the original,
    # and checks that the stored field is an immutable dictionary while the serialized one is a
    # plain dictionary.
    class Zoo(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a tuple of keeper names and a nested mapping of animal features.
      zookeepers: typing.Tuple[str, ...]
      animals: typing.Mapping[str, typing.Mapping[str, tensor.Tensor]]

    featurespec = {
        'size': tensor.TensorSpec([3]),
        'weight': tensor.TensorSpec([]),
    }
    zoo_spec = Zoo.Spec(
        zookeepers=['Zoey', 'Zack'],
        animals={'tiger': featurespec, 'elephant': featurespec},
    )

    serialized = zoo_spec._serialize()
    self.assertEqual(
        serialized,
        (
            ('zookeepers', ('Zoey', 'Zack')),
            ('animals', {'tiger': featurespec, 'elephant': featurespec}),
        ),
    )
    restored = Zoo.Spec._deserialize(serialized)
    self.assertEqual(zoo_spec, restored)

    # ImmutableDict is used for the field, but dict for the serialization:
    self.assertIsInstance(zoo_spec.animals, immutable_dict.ImmutableDict)
    serialized_field_name, serialized_field_value = serialized[1]
    self.assertEqual(serialized_field_name, 'animals')
    self.assertIsInstance(serialized_field_value, dict)

  def testSpecComponents(self):
    # [FUNCTION DOC]
    # Converts a zoo value to its flat component tensors via its spec (nested mappings ordered
    # by key), asserts five components with expected contents, rebuilds an equal value from
    # them, and asserts the spec's component specs have the expected shapes and types.
    class Zoo(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local fixture type with a tuple of keeper names and a nested mapping of animal features.
      zookeepers: typing.Tuple[str, ...]
      animals: typing.Mapping[str, typing.Mapping[str, tensor.Tensor]]

    zoo = Zoo(
        ['Zoey', 'Zack'],
        {
            'elephant': {'size': [25, 30, 20], 'weight': 2000.0},
            'tiger': {'hunger': 3.2, 'size': [3, 8, 2], 'weight': 87.3},
        },
    )
    zoo_spec = Zoo.Spec.from_value(zoo)

    components = zoo_spec._to_components(zoo)
    self.assertLen(components, 5)
    self.assertAllClose(components[0], [25, 30, 20])
    self.assertAllClose(components[1], 2000.0)
    self.assertAllClose(components[2], 3.2)
    self.assertAllClose(components[3], [3, 8, 2])
    self.assertAllClose(components[4], 87.3)

    restored = zoo_spec._from_components(components)
    self.assertAllEqual(zoo == restored, True)

    self.assertEqual(
        zoo_spec._component_specs,
        (
            tensor.TensorSpec([3], dtypes.int32),
            tensor.TensorSpec([], dtypes.float32),
            tensor.TensorSpec([], dtypes.float32),
            tensor.TensorSpec([3], dtypes.int32),
            tensor.TensorSpec([], dtypes.float32),
        ),
    )

  def testCopyAndPickle(self):
    # [FUNCTION DOC]
    # Asserts a spec equals its shallow copy, deep copy and a copy that was serialized with
    # pickle and loaded again.
    values_spec = tensor.TensorSpec([4], dtypes.float32)
    mask_spec = tensor.TensorSpec([4], dtypes.bool)
    mt_spec = MaskedTensorV1.Spec(values_spec, mask_spec)
    self.assertEqual(copy.copy(mt_spec), mt_spec)
    self.assertEqual(copy.deepcopy(mt_spec), mt_spec)
    self.assertEqual(pickle.loads(pickle.dumps(mt_spec)), mt_spec)

  def testCustomizeSpecTest(self):
    # [FUNCTION DOC]
    # Defines a type with a customized spec (own constructor from shape and dtype, validation,
    # properties and a with-shape method). Builds a value and its spec, asserts the spec's shape
    # and dtype, equality with a directly constructed spec, and that the with-shape method
    # yields the spec of a differently shaped value.
    class WeightedTensor(extension_type.ExtensionType):
      # [CLASS DOC]
      # KNOWN ISSUE: the shape and dtype properties on the value class itself read the same
      # attribute they define, so reading them on an instance would recurse endlessly; the
      # test only uses the spec's properties.
      # Local fixture type with values and scalar weight tensors, a validation hook requiring
      # a scalar weight, and a customized nested spec class.
      """ExtensionType with a customized TypeSpec.

      * Custom constructor.
      * Custom __validate__.
      * Add properties (shape, dtype, weight_dtype).
      * Add method (with_shape).
      """

      values: tensor.Tensor
      weight: tensor.Tensor  # scalar

      shape = property(lambda self: self.shape)
      dtype = property(lambda self: self.dtype)
      weight_dtype = property(lambda self: self.weight.dtype)

      def __validate__(self):
        # [FUNCTION DOC]
        # Validation hook: asserts the weight field has rank zero, raising a value error
        # otherwise. Returns nothing.
        self.weight.shape.assert_has_rank(0)

      class Spec:
        # [CLASS DOC]
        # Customized type specification: constructed from shape, dtype and optional weight
        # dtype instead of per-field specs; exposes shape, dtype and weight dtype properties
        # and a with-shape method.

        def __init__(self, shape, dtype, weight_dtype=dtypes.float32):
          # [FUNCTION DOC]
          # Custom spec constructor. Inputs: shape and dtype of the values, and optional weight
          # dtype (default 32-bit float). Stores a tensor spec for values and a scalar tensor
          # spec for weight. Returns nothing.
          self.values = tensor.TensorSpec(shape, dtype)
          self.weight = tensor.TensorSpec([], weight_dtype)

        def __validate__(self):
          # [FUNCTION DOC]
          # Validation hook: asserts the weight spec has rank zero, raising a value error
          # otherwise. Returns nothing.
          self.weight.shape.assert_has_rank(0)

        shape = property(lambda self: self.values.shape)
        dtype = property(lambda self: self.values.dtype)
        weight_dtype = property(lambda self: self.weight.dtype)

        def with_shape(self, shape):
          # [FUNCTION DOC]
          # Input: shape, the new values shape. Returns a new spec of this customized kind with
          # that shape and the same dtype and weight dtype.
          return WeightedTensor.Spec(shape, self.dtype, self.weight_dtype)

    wt = WeightedTensor([1, 2], 0.3)
    wt_spec = WeightedTensor.Spec.from_value(wt)
    self.assertEqual(wt_spec.shape, tensor_shape.TensorShape([2]))
    self.assertEqual(wt_spec.dtype, dtypes.int32)

    self.assertEqual(wt_spec, WeightedTensor.Spec([2], dtypes.int32))

    wt2 = WeightedTensor([[1, 2], [3, 4]], 0.5)
    wt2_spec = WeightedTensor.Spec.from_value(wt2)
    self.assertEqual(wt_spec.with_shape([2, 2]), wt2_spec)

  def testNestedSpecMustBeAClass(self):
    # [FUNCTION DOC]
    # Asserts that defining a type whose nested spec attribute is a number rather than a class
    # raises a value error saying the spec must be a nested class.
    with self.assertRaisesRegex(
        ValueError, r'BrokenExtensionType\.Spec must be a nested class; got 12.'
    ):

      class BrokenExtensionType(extension_type.ExtensionType):
        # [CLASS DOC]
        # Local type whose definition is expected to fail: its spec attribute is not a class.
        Spec = 12  # pylint: disable=invalid-name

      del BrokenExtensionType

  def testNestedSpecMayNotHaveBaseClasses(self):
    # [FUNCTION DOC]
    # Asserts that defining a type whose nested spec class explicitly derives from a type
    # specification base class raises a value error saying the spec must be directly derived
    # from the plain base type specification.
    with self.assertRaisesRegex(
        ValueError,
        r'BrokenExtensionType\.Spec must be directly subclassed '
        'from tf.TypeSpec.',
    ):

      class BrokenExtensionType(extension_type.ExtensionType):
        # [CLASS DOC]
        # Local type whose definition is expected to fail: nested spec has a base class.

        class Spec(type_spec.BatchableTypeSpec):
          # [CLASS DOC]
          # Invalid nested spec: explicitly derives from the batchable type specification.
          pass

      del BrokenExtensionType


@test_util.run_all_in_graph_and_eager_modes
class AnonymousExtensionTypeTest(
    test_util.TensorFlowTestCase, parameterized.TestCase
):
  # [CLASS DOC]
  # Tests for the schema-less extension type that holds arbitrary named fields: construction,
  # construction errors, attribute access and immutability, reinterpreting typed values as
  # anonymous ones, and related behavior. Runs in graph and eager modes.

  @parameterized.parameters([
      [dict(i=5, f=3.2, b=True, n=None)],
      [dict(x=(1, 2), y={3: 4, 5: 6})],
      [lambda: dict(t=constant_op.constant(123))],
      [lambda: dict(r=ragged_factory_ops.constant([[1, 2], [3]]))],
  ])
  def testConstruction(self, fields):
    # [FUNCTION DOC]
    # Parameterized test. Input: mapping of field names to values (or deferred builder when
    # tensors are involved). Asserts an anonymous extension type can be constructed from
    # scalars, nested tuples/mappings, tensors and ragged tensors without error.
    if callable(fields):
      fields = fields()
    extension_type.AnonymousExtensionType(**fields)

  @parameterized.parameters([
      [dict(x=[1, 2, 3]), 'unsupported `value` argument'],
      [dict(x=set([1, 2])), 'unsupported `value` argument'],
      [dict(x=(1, dict([(2, [])]))), 'unsupported `value` argument'],
      [
          dict(_tf_extension_type_xyz=5),
          'Reserved field name .*_tf_extension_type_xyz.*',
      ],
  ])
  def testConstructionErrors(self, fields, error):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: field mapping and expected message pattern. Asserts that
    # construction raises a value error for unsupported value kinds (list, set, mapping holding
    # a list) and for reserved field names.
    with self.assertRaisesRegex(ValueError, error):
      extension_type.AnonymousExtensionType(**fields)

  @parameterized.parameters([
      [dict(i=5, f=3.2, b=True, n=None)],
      [dict(x=(1, 2), y={3: 4, 5: 6})],
      [lambda: dict(t=constant_op.constant(123))],
      [lambda: dict(r=ragged_factory_ops.constant([[1, 2], [3]]))],
  ])
  def testAttributeAccessors(self, fields):
    # [FUNCTION DOC]
    # Parameterized test. Input: field mapping (or deferred builder). Builds an anonymous value
    # and asserts each field reads back equal to the supplied value (array comparison for
    # tensors).
    if callable(fields):
      fields = fields()
    s = extension_type.AnonymousExtensionType(**fields)
    for name, value in fields.items():
      actual = getattr(s, name)
      if isinstance(actual, (tensor.Tensor, ragged_tensor.RaggedTensor)):
        self.assertAllEqual(actual, value)
      else:
        self.assertEqual(actual, value)

  def testAttributeAccessorsAreImmutable(self):
    # [FUNCTION DOC]
    # Asserts that setting or deleting a field of an anonymous value raises an attribute error,
    # and that assigning an item in a nested mapping field raises a type error.
    s = extension_type.AnonymousExtensionType(x=12, y={'x': 55})
    with self.assertRaisesRegex(AttributeError, 'Cannot set attribute `x`'):
      s.x = 22
    with self.assertRaisesRegex(AttributeError, 'Cannot delete attribute `y`'):
      del s.y
    with self.assertRaisesRegex(TypeError, 'does not support item assignment'):
      s.y['x'] = 66

  def testReinterpret(self):
    # [FUNCTION DOC]
    # Reinterprets a richer masked tensor as an anonymous value, back to the same type, and as
    # the simpler masked tensor type; asserts values and mask are preserved in each case.
    x = MaskedTensorV2([4, 5], [True, False])
    anon_x = extension_type.reinterpret(
        x, extension_type.AnonymousExtensionType
    )
    self.assertAllEqual(anon_x.values, [4, 5])
    self.assertAllEqual(anon_x.mask, [True, False])

    round_trip_x = extension_type.reinterpret(anon_x, MaskedTensorV2)
    self.assertAllEqual(round_trip_x.values, [4, 5])
    self.assertAllEqual(round_trip_x.mask, [True, False])

    converted_x = extension_type.reinterpret(anon_x, MaskedTensorV1)
    self.assertAllEqual(converted_x.values, [4, 5])
    self.assertAllEqual(converted_x.mask, [True, False])

  # pylint: disable=g-long-lambda
  @parameterized.parameters([
      [
          lambda: extension_type.AnonymousExtensionType(
              values=constant_op.constant([1, 2, 3])
          ),
          MaskedTensorV2,
          "Missing required fields: {'mask'}",
      ],
      [
          lambda: extension_type.AnonymousExtensionType(
              values=(1, 2, 3), mask=None
          ),
          MaskedTensorV2,
          "mask: expected a Tensor, got 'NoneType'",
      ],
      [
          lambda: extension_type.AnonymousExtensionType(
              values=constant_op.constant([1, 2, 3]),
              mask=constant_op.constant([True, False]),
          ),
          MaskedTensorV2,
          'Shapes .* are incompatible',
      ],
      [
          lambda: extension_type.AnonymousExtensionType(
              values=constant_op.constant([1, 2, 3])
          ),
          tensor.Tensor,
          (
              'reinterpret expects `new_type` to be a subclass of '
              'tf.ExtensionType; '
              'got .*.Tensor.*'
          ),
      ],
      [
          lambda: constant_op.constant([1, 2, 3]),
          extension_type.AnonymousExtensionType,
          (
              'reinterpret expects `value` to be a tf.ExtensionType instance; '
              'got.*.Tensor.*'
          ),
      ],
  ])
  def testReinterpretErrors(self, value, new_type, error):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: a value (or deferred builder), target type and expected
    # message pattern. Asserts reinterpretation raises a type or value error with the message:
    # missing field, wrongly typed field, incompatible shapes, target not an extension type,
    # or source not an extension type value.
    if callable(value):
      value = value()
    with self.assertRaisesRegex((TypeError, ValueError), error):
      extension_type.reinterpret(value, new_type)

  def testLoadSavedModelWithUnregisteredExtensionType(self):
    # [FUNCTION DOC]
    # Saves a module containing a compiled function traced for all combinations of plain tensor
    # and masked-tensor spec arguments, with the spec temporarily registered, then loads it after
    # the registration is gone. Asserts the lookup by name fails and calling the loaded function
    # returns anonymous extension values with the expected sums and masks, which can be
    # reinterpreted as the original masked tensor type.
    def f(x, y):
      # [FUNCTION DOC]
      # Function saved in the model. Inputs: x and y, each a masked tensor or a plain tensor.
      # Adds the values (plain tensors count as fully unmasked), combines masks with logical
      # and, and returns a masked tensor.
      x_values = x.values if isinstance(x, MaskedTensorV1) else x
      y_values = y.values if isinstance(y, MaskedTensorV1) else y
      x_mask = x.mask if isinstance(x, MaskedTensorV1) else True
      y_mask = y.mask if isinstance(y, MaskedTensorV1) else True
      return MaskedTensorV1(x_values + y_values, x_mask & y_mask)

    t_spec = tensor.TensorSpec(None, dtypes.int32)
    b_spec = tensor.TensorSpec(None, dtypes.bool)
    mt_spec = MaskedTensorV1.Spec(values=t_spec, mask=b_spec)
    model = module.Module()
    model.f = def_function.function(f)
    model.f.get_concrete_function(t_spec, t_spec)
    model.f.get_concrete_function(t_spec, mt_spec)
    model.f.get_concrete_function(mt_spec, t_spec)
    model.f.get_concrete_function(mt_spec, mt_spec)

    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    with temporarily_register_type_spec(
        'tf.test.MaskedTensorV1.Spec', MaskedTensorV1.Spec
    ):
      save.save(model, path)
    loaded_model = load.load(path)

    with self.assertRaises(ValueError):
      type_spec_registry.lookup('tf.test.MaskedTensorV1')

    t = constant_op.constant([10, 20, 30])
    v1 = loaded_model.f(t, t)
    self.assertIsInstance(v1, extension_type.AnonymousExtensionType)
    self.assertAllEqual(v1.values, [20, 40, 60])
    self.assertAllEqual(v1.mask, True)

    v2 = loaded_model.f(v1, v1)
    self.assertIsInstance(v2, extension_type.AnonymousExtensionType)
    self.assertAllEqual(v2.values, [40, 80, 120])
    self.assertAllEqual(v2.mask, True)

    mt = MaskedTensorV1([1, 2, 3], [True, True, False])
    v3 = loaded_model.f(
        t, extension_type.reinterpret(mt, extension_type.AnonymousExtensionType)
    )
    self.assertIsInstance(v3, extension_type.AnonymousExtensionType)
    self.assertAllEqual(v3.values, [11, 22, 33])
    self.assertAllEqual(v3.mask, [True, True, False])

    v4 = extension_type.reinterpret(v3, MaskedTensorV1)
    self.assertIsInstance(v4, MaskedTensorV1)
    self.assertAllEqual(v4.values, [11, 22, 33])
    self.assertAllEqual(v4.mask, [True, True, False])

  def testFlatTensorSpecs(self):
    # [FUNCTION DOC]
    # Gets the spec of a masked tensor value and asserts its flat tensor specs are the two
    # expected unnamed tensor specs (integer values and boolean mask, shape of two).
    x = MaskedTensorV2([4, 5], [True, False])
    spec = type_spec.type_spec_from_value(x)
    flat_specs = spec._flat_tensor_specs
    self.assertEqual(
        flat_specs,
        [
            tensor.TensorSpec(shape=(2,), dtype=dtypes.int32, name=None),
            tensor.TensorSpec(shape=(2,), dtype=dtypes.bool, name=None),
        ],
    )

  def testFullTypesForFlatTensors(self):
    # [FUNCTION DOC]
    # Asks for the full type annotations of the flat tensors of a masked tensor's spec and
    # asserts one entry per flat tensor spec, each with the unset type identifier.
    x = MaskedTensorV2([4, 5], [True, False])
    spec = type_spec.type_spec_from_value(x)
    full_type_list = fulltypes_for_flat_tensors(spec)
    expect = [
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET),
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET),
    ]
    self.assertEqual(len(spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

  def testToLegacyOutputShape(self):
    # [FUNCTION DOC]
    # For the fixture type that customizes its spec with a shape property, asserts the legacy
    # output shape of its spec is the shape of the single tensor field (a vector of two).
    x = SimpleExtensionType([4, 5])
    spec = type_spec.type_spec_from_value(x)
    shape = spec._to_legacy_output_shapes()
    self.assertAllEqual(shape.as_list(), [2])

  def testToLegacyOutputShapeMissing(self):
    # [FUNCTION DOC]
    # For a masked tensor whose spec lacks a shape property, asserts asking for the legacy
    # output shape raises a not-implemented error.
    x = MaskedTensorV2([4, 5], [True, False])
    spec = type_spec.type_spec_from_value(x)
    with self.assertRaises(NotImplementedError):
      spec._to_legacy_output_shapes()


def replace_tensors_with_placeholders(value):
  # [FUNCTION DOC]
  # HELPER: shared test utility that erases static shape information. Input: value, any nested
  # structure (including composite tensors, which are expanded). Returns the same structure
  # with each dense tensor replaced by a placeholder that defaults to the tensor but has
  # unknown shape; other leaves are unchanged.
  def repl(x):
    # [FUNCTION DOC]
    # HELPER: leaf mapper for the placeholder replacement. Input: x, a leaf. Returns a
    # shapeless placeholder-with-default for a dense tensor, otherwise x unchanged.
    if isinstance(x, tensor.Tensor):
      return array_ops.placeholder_with_default(x, shape=None)
    else:
      return x

  return nest.map_structure(repl, value, expand_composites=True)


@contextlib.contextmanager
def temporarily_add_dispatch(op, typ, fn):
  # [FUNCTION DOC]
  # HELPER: context manager used by the operator overloading test. Inputs: op, an operation
  # with a fallback dispatcher list; typ, the type to dispatch for; fn, the dispatcher.
  # Registers the dispatcher for the duration of the with-block (global side effect on the
  # operation's dispatcher list), then asserts exactly one dispatcher was added and removes the
  # last one. Yields nothing.
  # KNOWN ISSUE: cleanup is not in a finally clause, so if the with-body raises, the dispatcher
  # stays registered and leaks into later tests.
  n = len(op._tf_fallback_dispatchers)
  dispatch.dispatch_for_types(op, typ)(fn)
  yield
  assert len(op._tf_fallback_dispatchers) == n + 1
  del op._tf_fallback_dispatchers[-1]


@contextlib.contextmanager
def temporarily_register_type_spec(name, cls):
  # [FUNCTION DOC]
  # HELPER: context manager for tests that need a type spec registered only temporarily.
  # Inputs: name, the registry name string; cls, the spec class. Registers the class under the
  # name in the global type spec registry, yields, then asserts the registry maps match and
  # removes both entries from the registry's internal tables. Mutates global registry state.
  # KNOWN ISSUE: cleanup is not in a finally clause, so a failure in the with-body leaves the
  # registration in place.
  """Context manager for making temporary changes to the TypeSpec registry."""
  type_spec_registry.register(name)(cls)
  yield
  assert type_spec_registry._TYPE_SPEC_TO_NAME.pop(cls) == name
  assert type_spec_registry._NAME_TO_TYPE_SPEC.pop(name) is cls


if __name__ == '__main__':
  googletest.main()
