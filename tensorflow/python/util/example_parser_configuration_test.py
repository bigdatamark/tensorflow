# [MODULE DOC]
# Test module for the example-parser configuration extraction: builds a real parse-example
# graph operation in a session and compares the extracted configuration with expected text
# format messages. Run directly as a test program. No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ExampleParserConfiguration."""

from google.protobuf import text_format

from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util.example_parser_configuration import extract_example_parser_configuration

EXPECTED_CONFIG_V1 = """
feature_map {
  key: "x"
  value {
    fixed_len_feature {
      dtype: DT_FLOAT
      shape {
        dim {
          size: 1
        }
      }
      default_value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 33.0
      }
      values_output_tensor_name: "ParseExample/ParseExample:3"
    }
  }
}
feature_map {
  key: "y"
  value {
    var_len_feature {
      dtype: DT_STRING
      values_output_tensor_name: "ParseExample/ParseExample:1"
      indices_output_tensor_name: "ParseExample/ParseExample:0"
      shapes_output_tensor_name: "ParseExample/ParseExample:2"
    }
  }
}
"""


EXPECTED_CONFIG_V2 = EXPECTED_CONFIG_V1.replace(
    'ParseExample/ParseExample:', 'ParseExample/ParseExampleV2:')


class ExampleParserConfigurationTest(test.TestCase):
  # [CLASS DOC]
  # Tests extract_example_parser_configuration against the real parse-example graph operation,
  # comparing the extracted configuration with a text-format expected message (with output
  # tensor names adjusted for the original or version-2 operation).

  def getExpectedConfig(self, op_type):
    # [FUNCTION DOC]
    # HELPER: builds the expected configuration for testBasic.
    # Input: op_type, the operation type name. Returns a configuration message parsed from the
    # version-2 expected text when the type is the version-2 kind, otherwise from the original text.
    expected = example_parser_configuration_pb2.ExampleParserConfiguration()
    if op_type == 'ParseExampleV2':
      text_format.Parse(EXPECTED_CONFIG_V2, expected)
    else:
      text_format.Parse(EXPECTED_CONFIG_V1, expected)
    return expected

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: in a session, parse a one-element string placeholder with a fixed-length float
    # feature (shape of one, default 33.0) and a variable-length string feature, then extract the
    # configuration from the operation behind the dense result.
    # Asserts the extracted message equals the expected message for that operation type.
    with session.Session() as sess:
      examples = array_ops.placeholder(dtypes.string, shape=[1])
      feature_to_type = {
          'x': parsing_ops.FixedLenFeature([1], dtypes.float32, 33.0),
          'y': parsing_ops.VarLenFeature(dtypes.string)
      }
      result = parsing_ops.parse_example(examples, feature_to_type)
      parse_example_op = result['x'].op
      config = extract_example_parser_configuration(parse_example_op, sess)
      expected = self.getExpectedConfig(parse_example_op.type)
      self.assertProtoEquals(expected, config)


if __name__ == '__main__':
  test.main()
