# [MODULE DOC]
# Unit tests for the nested structure coder: round trips of lists, tuples, named tuples, dicts,
# scalars, shapes, data types, tensor/ragged/sparse/bounded/dataset/extension type specs, tensors
# and numpy arrays through structured-value protos, plus unencodable, custom codec and type spec
# registration cases. Defines two trivial type spec helper classes (one registered at import).
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
"""Tests for nested structure coding."""

import collections
import typing
import warnings

import numpy as np

from google.protobuf import text_format
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal


class NestedStructureCoderTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the nested structure coder. Most tests encode a structure, check it is
  # encodable, compare the produced proto with an expected proto built by hand or from text
  # format, then decode and compare with the original. Also covers error and registration cases.
  # Uses two module-level trivial type spec classes (registered and unregistered).

  def testEncodeDecodeList(self):
    # [FUNCTION DOC]
    # Scenario: a list of three floats. Asserts it is encodable, the proto is a list value with
    # three 64-bit float entries, and decoding returns an equal list.
    structure = [1.5, 2.5, 3.0]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().float64_value = 1.5
    expected.list_value.values.add().float64_value = 2.5
    expected.list_value.values.add().float64_value = 3.0
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTuple(self):
    # [FUNCTION DOC]
    # Scenario: a tuple holding a string and a list that contains an integer and a nested tuple
    # of two integers. Asserts the expected nested proto and a round trip to an equal tuple.
    structure = ("hello", [3, (2, 1)])
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.tuple_value.values.add().string_value = "hello"
    list_value = expected.tuple_value.values.add().list_value
    list_value.values.add().int64_value = 3
    tuple_value = list_value.values.add().tuple_value
    tuple_value.values.add().int64_value = 2
    tuple_value.values.add().int64_value = 1
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeDict(self):
    # [FUNCTION DOC]
    # Scenario: a dict with an integer and a list of an integer and a float. Asserts the expected
    # dict proto, that the decoded integer is a Python integer, and equality after round trip.
    structure = dict(a=3, b=[7, 2.5])
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.dict_value.fields["a"].int64_value = 3
    list_value = expected.dict_value.fields["b"].list_value
    list_value.values.add().int64_value = 7
    list_value.values.add().float64_value = 2.5
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertIsInstance(decoded["a"], int)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeImmutableDict(self):
    # [FUNCTION DOC]
    # Scenario: same content as the dict test but wrapped in an immutable dictionary type, which
    # is a mapping. Asserts the same dict proto is produced and the decoded plain dict compares
    # equal to the immutable original.
    structure = immutable_dict.ImmutableDict(dict(a=3, b=[7, 2.5]))
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.dict_value.fields["a"].int64_value = 3
    list_value = expected.dict_value.fields["b"].list_value
    list_value.values.add().int64_value = 7
    list_value.values.add().float64_value = 2.5
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertIsInstance(decoded["a"], int)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTensorShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the last expected-value step is a chained assignment that rebinds the
    # expected shape variable to the string, which is harmless but misleading; it does set the
    # string entry as intended.
    # Scenario: a list with a known tensor shape and a string. Asserts the proto has a tensor
    # shape value with three dimensions followed by the string, and a round trip equality.
    structure = [tensor_shape.TensorShape([1, 2, 3]), "hello"]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_shape = expected_list.values.add().tensor_shape_value
    expected_tensor_shape.dim.add().size = 1
    expected_tensor_shape.dim.add().size = 2
    expected_tensor_shape.dim.add().size = 3
    expected_tensor_shape = expected_list.values.add().string_value = "hello"
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeNamedTuple(self):
    # [FUNCTION DOC]
    # Scenario: a named tuple with a list field and a string field. Asserts the named tuple
    # proto (type name plus ordered key/value pairs), and that the decoded value has the same
    # fields mapping and class name (though it is a newly created type).
    named_tuple_type = collections.namedtuple("NamedTuple", ["x", "y"])
    named_tuple = named_tuple_type(x=[1, 2], y="hello")
    self.assertTrue(nested_structure_coder.can_encode(named_tuple))
    encoded = nested_structure_coder.encode_structure(named_tuple)
    expected = struct_pb2.StructuredValue()
    expected_named_tuple = expected.named_tuple_value
    expected_named_tuple.name = "NamedTuple"
    key_value_pair = expected_named_tuple.values.add()
    key_value_pair.key = "x"
    list_value = key_value_pair.value.list_value
    list_value.values.add().int64_value = 1
    list_value.values.add().int64_value = 2
    key_value_pair = expected_named_tuple.values.add()
    key_value_pair.key = "y"
    key_value_pair.value.string_value = "hello"
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(named_tuple._asdict(), decoded._asdict())
    self.assertEqual(named_tuple.__class__.__name__, decoded.__class__.__name__)

  def testNone(self):
    # [FUNCTION DOC]
    # Scenario: a list containing a float and the null value. Asserts the null is encoded as an
    # empty none value message and the round trip restores the list.
    structure = [1.0, None]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().float64_value = 1.0
    expected.list_value.values.add().none_value.CopyFrom(struct_pb2.NoneValue())
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testBool(self):
    # [FUNCTION DOC]
    # Scenario: a list with one false boolean. Asserts it is encoded in the boolean field (not as
    # an integer) and round trips.
    structure = [False]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().bool_value = False
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEmptyStructures(self):
    # [FUNCTION DOC]
    # Scenario: a list of an empty list, an empty dict and an empty tuple. Asserts each is
    # encoded as its own present-but-empty message kind and the round trip is equal.
    structure = [list(), dict(), tuple()]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected.list_value.values.add().list_value.CopyFrom(struct_pb2.ListValue())
    expected.list_value.values.add().dict_value.CopyFrom(struct_pb2.DictValue())
    expected.list_value.values.add().tuple_value.CopyFrom(
        struct_pb2.TupleValue())
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testDtype(self):
    # [FUNCTION DOC]
    # Scenario: a list holding the 64-bit integer data type. Asserts it is stored as the data
    # type enum number in the tensor dtype field and round trips.
    structure = [dtypes.int64]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    list_value = expected.list_value.values.add()
    list_value.tensor_dtype_value = dtypes.int64.as_datatype_enum
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTensorSpec(self):
    # [FUNCTION DOC]
    # Scenario: a list with a named tensor spec (shape 1 by 2 by 3, 64-bit integer). Asserts the
    # tensor spec message has the dimensions, name and data type, and that it round trips.
    structure = [tensor.TensorSpec([1, 2, 3], dtypes.int64, "hello")]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 1
    expected_tensor_spec.shape.dim.add().size = 2
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = "hello"
    expected_tensor_spec.dtype = dtypes.int64.as_datatype_enum
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeTensorSpecWithNoName(self):
    # [FUNCTION DOC]
    # Scenario: like the tensor spec test but with no name given. Asserts the encoded name is
    # the empty string and the spec round trips.
    structure = [tensor.TensorSpec([1, 2, 3], dtypes.int64)]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 1
    expected_tensor_spec.shape.dim.add().size = 2
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = ""
    expected_tensor_spec.dtype = dtypes.int64.as_datatype_enum
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeRaggedTensorSpec(self):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor spec (shape 1 by 2 by 3, 64-bit integer values, ragged rank 2,
    # 32-bit row splits). Parses an expected text-format proto and asserts the encoding has the
    # ragged spec class, name, three flat components and a state tuple of shape, data type,
    # rank and row splits type; then asserts the round trip is equal.
    structure = [
        ragged_tensor.RaggedTensorSpec([1, 2, 3], dtypes.int64, 2, dtypes.int32)
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
      list_value {
        values {
          type_spec_value {
            type_spec_class: RAGGED_TENSOR_SPEC
            type_spec_class_name: 'RaggedTensorSpec'
            num_flat_components: 3
            type_state {
              tuple_value {
                # spec._shape
                values {
                  tensor_shape_value {
                    dim { size: 1 }
                    dim { size: 2 }
                    dim { size: 3 }
                  }
                }
                # spec._dtype
                values { tensor_dtype_value: DT_INT64 }
                # spec._ragged_rank
                values { int64_value: 2 }
                # spec._row_splits_dtype
                values { tensor_dtype_value: DT_INT32 }
              }
            }
          }
        }
      }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeSparseTensorSpec(self):
    # [FUNCTION DOC]
    # Scenario: a sparse tensor spec of shape 10 by 20 and 32-bit float. Asserts the text-format
    # expected proto (sparse spec class, three flat components, state of shape and data type)
    # and a round trip.
    structure = [sparse_tensor.SparseTensorSpec([10, 20], dtypes.float32)]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
      list_value {
        values {
          type_spec_value {
            type_spec_class: SPARSE_TENSOR_SPEC
            type_spec_class_name: 'SparseTensorSpec'
            num_flat_components: 3
            type_state {
              tuple_value {
                # spec._shape
                values {
                  tensor_shape_value {
                    dim { size: 10 }
                    dim { size: 20 }
                  }
                }
                # spec._dtype
                values { tensor_dtype_value: DT_FLOAT }
              }
            }
          }
        }
      }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeExtensionTypeSpec(self):
    # [FUNCTION DOC]
    # Scenario: defines a local extension type Zoo (tuple of keeper names, mapping of animals to
    # tensors) and encodes a list holding its spec. Asserts the text-format expected proto
    # (extension type spec class, registered-style name ending in Spec, one flat component,
    # nested state) and that decoding gives an equal spec.

    class Zoo(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local extension type used only to create a spec for the test: has a tuple of strings
      # field named zookeepers and a mapping field named animals from strings to tensors.
      __name__ = "tf.nested_structure_coder_test.Zoo"
      zookeepers: typing.Tuple[str, ...]
      animals: typing.Mapping[str, tensor.Tensor]

    structure = [
        Zoo.Spec(
            zookeepers=["Zoey", "Zack"],
            animals={"tiger": tensor.TensorSpec([16])})
    ]

    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
    list_value {
      values {
        type_spec_value {
          type_spec_class: EXTENSION_TYPE_SPEC
          type_spec_class_name: "tf.nested_structure_coder_test.Zoo.Spec"
          num_flat_components: 1
          type_state {
            tuple_value {
              values {
                tuple_value {
                  values { string_value: "zookeepers" }
                  values { tuple_value {
                    values { string_value: "Zoey" }
                    values { string_value: "Zack" } } } } }
              values {
                tuple_value {
                  values { string_value: "animals" }
                  values { dict_value {
                    fields {
                      key: "tiger"
                      value { tensor_spec_value {
                        shape { dim { size: 16 } }
                        dtype: DT_FLOAT } } } } } } } } } } } }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testDecodeUnknownTensorSpec(self):
    # [FUNCTION DOC]
    # Scenario: a hand-built proto with a type spec message of unknown class number zero and the
    # name FutureTensorSpec. Asserts decoding raises a value error saying the type is not
    # supported by this version.
    encoded = struct_pb2.StructuredValue()
    encoded.type_spec_value.type_spec_class = 0
    encoded.type_spec_value.type_spec_class_name = "FutureTensorSpec"
    with self.assertRaisesRegex(ValueError,
                                "The type 'FutureTensorSpec' is not supported"):
      nested_structure_coder.decode_proto(encoded)

  def testEncodeDecodeBoundedTensorSpec(self):
    # [FUNCTION DOC]
    # Scenario: a bounded tensor spec (shape 1 by 2 by 3, 64-bit integer, minimum 0, maximum 10,
    # name given). Asserts the bounded spec message including minimum and maximum as tensor
    # protos, and a round trip.
    structure = [
        tensor.BoundedTensorSpec([1, 2, 3], dtypes.int64, 0, 10, "hello_0_10")
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().bounded_tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 1
    expected_tensor_spec.shape.dim.add().size = 2
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = "hello_0_10"
    expected_tensor_spec.dtype = dtypes.int64.as_datatype_enum
    expected_tensor_spec.minimum.CopyFrom(
        tensor_util.make_tensor_proto([0], dtype=dtypes.int64, shape=[]))
    expected_tensor_spec.maximum.CopyFrom(
        tensor_util.make_tensor_proto([10], dtype=dtypes.int64, shape=[]))
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDecodeBoundedTensorSpecNoName(self):
    # [FUNCTION DOC]
    # Scenario: a bounded 64-bit float spec of shape 28 by 28 by 3 with scalar minimum and a
    # per-channel maximum of three values, no name. Asserts an empty name, the bounds as tensor
    # protos, and a round trip.
    structure = [
        tensor.BoundedTensorSpec((28, 28, 3), dtypes.float64, -2, (1, 1, 20))
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected = struct_pb2.StructuredValue()
    expected_list = expected.list_value
    expected_tensor_spec = expected_list.values.add().bounded_tensor_spec_value
    expected_tensor_spec.shape.dim.add().size = 28
    expected_tensor_spec.shape.dim.add().size = 28
    expected_tensor_spec.shape.dim.add().size = 3
    expected_tensor_spec.name = ""
    expected_tensor_spec.dtype = dtypes.float64.as_datatype_enum
    expected_tensor_spec.minimum.CopyFrom(
        tensor_util.make_tensor_proto([-2], dtype=dtypes.float64, shape=[]))
    expected_tensor_spec.maximum.CopyFrom(
        tensor_util.make_tensor_proto([1, 1, 20],
                                      dtype=dtypes.float64,
                                      shape=[3]))
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testEncodeDataSetSpec(self):
    # [FUNCTION DOC]
    # Scenario: a dataset spec whose element structure is a mapping of a ragged spec, a sparse
    # spec and a string tensor spec. Asserts only encodability and that the decoded structure
    # equals the original (the proto layout is not checked).
    structure = [
        dataset_ops.DatasetSpec({
            "rt": ragged_tensor.RaggedTensorSpec([10, None], dtypes.int32),
            "st": sparse_tensor.SparseTensorSpec([10, 20], dtypes.float32),
            "t": tensor.TensorSpec([10, 8], dtypes.string)
        })
    ]
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  @test_util.run_in_graph_and_eager_modes
  def testEncodeDecodeTensor(self):
    # [FUNCTION DOC]
    # Scenario (run in both graph and eager modes): a scalar integer constant. Asserts it is
    # encodable as a tensor value proto (32-bit integer, scalar shape, value one) and that the
    # decoded result has equal contents.
    structure = constant_op.constant(1)
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
      tensor_value {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertAllEqual(structure, decoded)

  def testEncodeDecodeNumpy(self):
    # [FUNCTION DOC]
    # Scenario: a zero-dimensional numpy array of the float one. Asserts the numpy value proto
    # (double data type, scalar shape, value 1.0), that decoding returns a numpy array, and
    # equal contents.
    structure = np.array(1.0)
    self.assertTrue(nested_structure_coder.can_encode(structure))
    encoded = nested_structure_coder.encode_structure(structure)
    expected_pbtxt = r"""
      numpy_value {
        dtype: DT_DOUBLE
        tensor_shape {
        }
        double_val: 1.0
      }
    """
    expected = struct_pb2.StructuredValue()
    text_format.Parse(expected_pbtxt, expected)
    self.assertEqual(expected, encoded)

    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertIsInstance(decoded, np.ndarray)
    self.assertAllEqual(structure, decoded)

  def testNotEncodable(self):
    # [FUNCTION DOC]
    # Scenario: a list containing an instance of a plain local class with no codec. Asserts
    # that the encodability check returns false.

    class NotEncodable(object):
      # [CLASS DOC]
      # Empty local class used as an object that no codec handles.
      pass

    self.assertFalse(nested_structure_coder.can_encode([NotEncodable()]))

  def testRegisterCustomCodec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: registers a codec in the global codec list and never removes it, so the
    # custom codec stays active for every later test in the same process.
    # Scenario: defines a local class and a codec that accepts it for encoding (returning an
    # empty structured value) and never for decoding, registers the codec, and asserts an
    # instance of the class is now reported as encodable.

    class MyObject(object):
      # [CLASS DOC]
      # Empty local class used as the custom type handled by the test codec.
      pass

    class MyObjectCodec(object):
      # [CLASS DOC]
      # Minimal custom codec for the local MyObject class, implementing the four codec
      # operations. It encodes to an empty proto and refuses to decode anything.
      """Codec for MyObject."""

      def can_encode(self, pyobj):
        # [FUNCTION DOC]
        # Returns true if pyobj is an instance of the local MyObject class.
        return isinstance(pyobj, MyObject)

      def do_encode(self, array, encode_fn):
        # [FUNCTION DOC]
        # Ignores both arguments and returns an empty structured value.
        del array, encode_fn
        return struct_pb2.StructuredValue()

      def can_decode(self, value):
        # [FUNCTION DOC]
        # Always returns false: this codec never decodes.
        del value
        return False

      def do_decode(self, value, decode_fn):
        # [FUNCTION DOC]
        # Always raises a not-implemented error; it is never reached because decoding is
        # refused by the can-decode check.
        raise NotImplementedError("Test only.")

    nested_structure_coder.register_codec(MyObjectCodec())
    my_object = MyObject()
    self.assertTrue(nested_structure_coder.can_encode(my_object))

  def testRegisteredTypeSpec(self):
    # [FUNCTION DOC]
    # Scenario: a mapping holding an instance of the registered trivial type spec. Asserts the
    # structure is encodable, that encoding emits exactly one warning mentioning the registered
    # name and the need to import and register the type, and that decoding gives an equal value.
    expected_warning = ("Encoding a StructuredValue with type "
                        "NestedStructureTest.RegisteredTypeSpec; loading "
                        "this StructuredValue will require that this type "
                        "be imported and registered")
    structure = {"x": RegisteredTypeSpec()}

    self.assertTrue(nested_structure_coder.can_encode(structure))
    with warnings.catch_warnings(record=True) as w:
      encoded = nested_structure_coder.encode_structure(structure)
      self.assertLen(w, 1)
      self.assertIn(expected_warning, str(w[0].message))
    decoded = nested_structure_coder.decode_proto(encoded)
    self.assertEqual(structure, decoded)

  def testUnregisteredTypeSpec(self):
    # [FUNCTION DOC]
    # Scenario: a mapping holding an instance of the unregistered trivial type spec. Asserts the
    # encodability check is false and encoding raises the not-encodable error.
    structure = {"x": UnregisteredTypeSpec()}
    self.assertFalse(nested_structure_coder.can_encode(structure))
    with self.assertRaises(nested_structure_coder.NotEncodableError):
      nested_structure_coder.encode_structure(structure)

  def testBuiltInTypeSpecCodecInvalidInputs(self):
    # [FUNCTION DOC]
    # Scenario: constructs the built-in type spec codec with invalid inputs and asserts the
    # value error messages: a class that is not a type spec; a type spec class that already has
    # a codec (dataset spec); an enum already in use; and enum values 0, 11, 12 and 13 reported
    # as invalid. Values 12 and 13 add no new coverage beyond 11 (same range check).

    class Foo:
      # [CLASS DOC]
      # Empty local class that is not a type spec; used to trigger the subclass check.
      pass

    class Bar(internal.TypeSpec):
      # [CLASS DOC]
      # Empty local type spec subclass with no codec yet; used to trigger the enum checks.
      pass

    with self.assertRaisesRegex(
        ValueError, "The type '(.*?)' does not subclass tf.TypeSpec."):
      nested_structure_coder.BuiltInTypeSpecCodec(Foo, 0)
    with self.assertRaisesRegex(
        ValueError, "The type '(.*?)' already has an instantiated codec."):
      nested_structure_coder.BuiltInTypeSpecCodec(
          dataset_ops.DatasetSpec, struct_pb2.TypeSpecProto.DATA_DATASET_SPEC)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is already registered."):
      nested_structure_coder.BuiltInTypeSpecCodec(
          Bar, struct_pb2.TypeSpecProto.DATA_DATASET_SPEC)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 0)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 11)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 12)
    with self.assertRaisesRegex(
        ValueError, "The proto value '(.*?)' is invalid."):
      nested_structure_coder.BuiltInTypeSpecCodec(Bar, 13)


# Trivial TypeSpec class for testing.
class UnregisteredTypeSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # HELPER: trivial type spec with no components and empty state, deliberately not registered
  # in the type spec registry; used to show it cannot be encoded.
  value_type = property(lambda self: None)
  _component_specs = property(lambda self: ())
  _to_components = lambda self, v: ()
  _from_components = classmethod(lambda cls, c: cls())
  _serialize = lambda self: ()


# Trivial TypeSpec class for testing.
@type_spec_registry.register("NestedStructureTest.RegisteredTypeSpec")
class RegisteredTypeSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # HELPER: trivial type spec with no components and empty state, registered at import time in
  # the type spec registry under the name NestedStructureTest.RegisteredTypeSpec; used to test
  # the generic registered type spec codec.
  value_type = property(lambda self: None)
  _component_specs = property(lambda self: ())
  _to_components = lambda self, v: ()
  _from_components = classmethod(lambda cls, c: cls())
  _serialize = lambda self: ()


if __name__ == "__main__":
  test.main()
