# [MODULE DOC]
# Unit tests for ConcreteFunction: parsing of user supplied function attributes into attribute
# protos (scalars, lists, invalid values) and rebuilding a ConcreteFunction from an atomic
# function or from a function definition plus function type, then calling it with nested
# structures of tensors, variables, ragged tensors and Python numbers.
# Runs with eager execution enabled when executed as a script.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ConcreteFunctionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering ConcreteFunction attribute handling and reconstruction from atomic
  # functions or function definitions. Parameterized for the scalar attribute cases. Shares
  # one helper that builds an empty function graph based concrete function with given attributes.

  def concrete_function_with_attrs(self, attrs):
    # [FUNCTION DOC]
    # HELPER: fixture for the attribute tests.
    # Creates an empty function graph named f and builds a concrete function from it with no
    # structured signature and the given mapping of attribute names to Python values. Returns
    # the concrete function. Raises whatever attribute conversion raises for invalid values.
    func_graph = func_graph_module.FuncGraph("f")
    return cf.ConcreteFunction.from_func_graph(func_graph, None, attrs=attrs)

  @parameterized.parameters(
      ({"api_implements": True}, attr_value_pb2.AttrValue(b=True)),
      ({"api_implements": 1}, attr_value_pb2.AttrValue(i=1)),
      ({"api_implements": 1.0}, attr_value_pb2.AttrValue(f=1.0)),
      (
          {"api_implements": "test"},
          attr_value_pb2.AttrValue(s=compat.as_bytes("test")),
      ),
  )
  def test_parses_func_attr_scalar_values(self, attrs, expected):
    # [FUNCTION DOC]
    # Parameterized over boolean, integer, float and string attribute values. Builds a concrete
    # function with a single attribute and asserts the stored function definition attribute
    # equals the expected attribute proto of the matching scalar kind (strings as bytes).
    self.assertEqual(
        self.concrete_function_with_attrs(attrs=attrs).function_def.attr[
            "api_implements"
        ],
        expected,
    )

  def test_parses_func_attr_list_values(self):
    # [FUNCTION DOC]
    # Builds a concrete function whose attribute is a list of a string, boolean, integer and
    # float, and asserts the stored attribute proto is a list value with each element in the
    # matching typed field.
    self.assertProtoEquals(
        r"""
        list {
            s: 'test'
            b: True
            i: 1
            f: 1.0
        }
        """,
        self.concrete_function_with_attrs(
            attrs={"api_implements": ["test", True, 1, 1.0]}
        ).function_def.attr["api_implements"],
    )

  def test_raises_value_error_for_invalid_attr(self):
    # [FUNCTION DOC]
    # Passing a null attribute value must raise a value error whose message says the attribute
    # must be of a supported type.
    with self.assertRaisesRegex(ValueError, "Attribute api_implements must be"):
      self.concrete_function_with_attrs(attrs={"api_implements": None})

  def test_generate_from_atomic(self):
    # [FUNCTION DOC]
    # Traces a function adding two dictionaries of tensor, variable, ragged tensor and Python
    # int values, keeps only its inference (atomic) function, deletes the originals, rebuilds a
    # concrete function from the atomic function, calls it and asserts every element sum
    # (including ragged flat values) is correct and the Python int comes back as a tensor.
    @polymorphic_function.function
    def add_dicts(dict_a, dict_b):
      # [FUNCTION DOC]
      # Traced function: returns a dictionary with, for each key of the first dictionary, the
      # sum of the two corresponding input values.
      result = {}
      for key in dict_a.keys():
        result[key] = dict_a[key] + dict_b[key]
      return result

    dict_a = {
        "tensor": constant_op.constant(1),
        "variable": variables.Variable(2),
        "ragged_tensor": ragged_tensor.RaggedTensor.from_row_splits(
            values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8]
        ),
        "python_int": 4,
    }
    dict_b = {
        "tensor": constant_op.constant(2),
        "variable": variables.Variable(5),
        "ragged_tensor": ragged_tensor.RaggedTensor.from_row_splits(
            values=[4, 2, 4, 1, 6, 9, 3, 6], row_splits=[0, 4, 4, 7, 8, 8]
        ),
        "python_int": 5,
    }

    original_concrete_fn = add_dicts.get_concrete_function(dict_a, dict_b)

    # Get the atomic function and delete everything else.
    atomic_fn = original_concrete_fn._inference_function
    del add_dicts
    del original_concrete_fn

    # Regenerate the ConcreteFunction.
    concrete_fn = cf.ConcreteFunction(atomic_fn)
    result = concrete_fn(dict_a, dict_b)

    # Call and check results.
    self.assertEqual(result["tensor"].numpy(), 3)
    self.assertEqual(result["variable"].numpy(), 7)
    self.assertEqual(
        result["ragged_tensor"].flat_values.numpy().tolist(),
        [7, 3, 8, 2, 11, 18, 5, 12],
    )
    self.assertEqual(result["python_int"].numpy(), 9)

  def test_generate_from_def(self):
    # [FUNCTION DOC]
    # Like the atomic test but without ragged tensors: traces the dictionary adding function,
    # keeps only its function definition and function type, deletes the originals, rebuilds an
    # atomic function from them, wraps it in a concrete function, calls it and asserts the
    # tensor, variable and Python int sums.
    @polymorphic_function.function
    def add_dicts(dict_a, dict_b):
      # [FUNCTION DOC]
      # Traced function: returns a dictionary with, for each key of the first dictionary, the
      # sum of the two corresponding input values.
      result = {}
      for key in dict_a.keys():
        result[key] = dict_a[key] + dict_b[key]
      return result

    dict_a = {
        "tensor": constant_op.constant(1),
        "variable": variables.Variable(2),
        "python_int": 4,
    }
    dict_b = {
        "tensor": constant_op.constant(2),
        "variable": variables.Variable(5),
        "python_int": 5,
    }

    original_concrete_fn = add_dicts.get_concrete_function(dict_a, dict_b)

    # Get FunctionDef + FunctionType and delete everything else.
    function_def = original_concrete_fn.function_def
    function_type = original_concrete_fn.function_type
    del add_dicts
    del original_concrete_fn

    # Regenerate the ConcreteFunction.
    atomic_fn = atomic_function.from_function_def(function_def, function_type)
    concrete_fn = cf.ConcreteFunction(atomic_fn)
    result = concrete_fn(dict_a, dict_b)

    # Call and check results.
    self.assertEqual(result["tensor"].numpy(), 3)
    self.assertEqual(result["variable"].numpy(), 7)
    self.assertEqual(result["python_int"].numpy(), 9)

if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
