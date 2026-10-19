# [MODULE DOC]
# Unit tests for the TFLite schema upgrade tool (upgrade_schema.Converter). Defines sample model
# dictionaries for schema versions 0 to 3 (empty, full operator rename set, and tensor buffer
# cases) and tests that upgrading them through temporary JSON, .bin and .tflite files yields the
# expected version 3 dictionaries. Requires the flatc compiler. No import-time side effects other
# than building the sample data.
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
"""Testing for updating TensorFlow lite schema."""

import json
import tempfile
from tensorflow.lite.schema import upgrade_schema as upgrade_schema_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib

EMPTY_TEST_SCHEMA_V1 = {
    "version": 1,
    "operator_codes": [],
    "subgraphs": [],
}

EMPTY_TEST_SCHEMA_V3 = {
    "version": 3,
    "operator_codes": [],
    "subgraphs": [],
    "buffers": [{
        "data": []
    }]
}

TEST_SCHEMA_V0 = {
    "operator_codes": [],
    "tensors": [],
    "inputs": [],
    "outputs": [],
    "operators": [],
    "version": 0
}

TEST_SCHEMA_V3 = {
    "operator_codes": [],
    "buffers": [{
        "data": []
    }],
    "subgraphs": [{
        "tensors": [],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "version":
        3
}

FULL_TEST_SCHEMA_V1 = {
    "version":
        1,
    "operator_codes": [
        {
            "builtin_code": "CONVOLUTION"
        },
        {
            "builtin_code": "DEPTHWISE_CONVOLUTION"
        },
        {
            "builtin_code": "AVERAGE_POOL"
        },
        {
            "builtin_code": "MAX_POOL"
        },
        {
            "builtin_code": "L2_POOL"
        },
        {
            "builtin_code": "SIGMOID"
        },
        {
            "builtin_code": "L2NORM"
        },
        {
            "builtin_code": "LOCAL_RESPONSE_NORM"
        },
        {
            "builtin_code": "ADD"
        },
        {
            "builtin_code": "Basic_RNN"
        },
    ],
    "subgraphs": [{
        "operators": [
            {
                "builtin_options_type": "PoolOptions"
            },
            {
                "builtin_options_type": "DepthwiseConvolutionOptions"
            },
            {
                "builtin_options_type": "ConvolutionOptions"
            },
            {
                "builtin_options_type": "LocalResponseNormOptions"
            },
            {
                "builtin_options_type": "BasicRNNOptions"
            },
        ],
    }],
    "description":
        "",
}

FULL_TEST_SCHEMA_V3 = {
    "version":
        3,
    "operator_codes": [
        {
            "builtin_code": "CONV_2D"
        },
        {
            "builtin_code": "DEPTHWISE_CONV_2D"
        },
        {
            "builtin_code": "AVERAGE_POOL_2D"
        },
        {
            "builtin_code": "MAX_POOL_2D"
        },
        {
            "builtin_code": "L2_POOL_2D"
        },
        {
            "builtin_code": "LOGISTIC"
        },
        {
            "builtin_code": "L2_NORMALIZATION"
        },
        {
            "builtin_code": "LOCAL_RESPONSE_NORMALIZATION"
        },
        {
            "builtin_code": "ADD"
        },
        {
            "builtin_code": "RNN"
        },
    ],
    "subgraphs": [{
        "operators": [
            {
                "builtin_options_type": "Pool2DOptions"
            },
            {
                "builtin_options_type": "DepthwiseConv2DOptions"
            },
            {
                "builtin_options_type": "Conv2DOptions"
            },
            {
                "builtin_options_type": "LocalResponseNormalizationOptions"
            },
            {
                "builtin_options_type": "RNNOptions"
            },
        ],
    }],
    "description":
        "",
    "buffers": [{
        "data": []
    }]
}

BUFFER_TEST_V2 = {
    "operator_codes": [],
    "buffers": [],
    "subgraphs": [{
        "tensors": [
            {
                "data_buffer": [1, 2, 3, 4]
            },
            {
                "data_buffer": [1, 2, 3, 4, 5, 6, 7, 8]
            },
            {
                "data_buffer": []
            },
        ],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "version":
        2
}

BUFFER_TEST_V3 = {
    "operator_codes": [],
    "subgraphs": [{
        "tensors": [
            {
                "buffer": 1
            },
            {
                "buffer": 2
            },
            {
                "buffer": 0
            },
        ],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "buffers": [
        {
            "data": []
        },
        {
            "data": [1, 2, 3, 4]
        },
        {
            "data": [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ],
    "version":
        3
}


def JsonDumpAndFlush(data, fp):
  # [FUNCTION DOC]
  # HELPER: small shared utility for the tests.
  # Serializes the dictionary data as JSON into the open file-like object fp and flushes it so
  # other readers see the contents. Returns nothing.
  """Write the dictionary `data` to a JSON file `fp` (and flush).

  Args:
    data: in a dictionary that is JSON serializable.
    fp: File-like object
  """
  json.dump(data, fp)
  fp.flush()


class TestSchemaUpgrade(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering upgrade_schema.Converter: error handling for bad files and extensions, and
  # correct upgrade of version 1 and 2 models to version 3. Has no shared fixtures; each test
  # builds its own converter and temporary files.

  def testNonExistentFile(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the temporary file made by mkstemp actually exists (it is empty), so the
    # expected error comes from the empty file/JSON handling path rather than a truly missing file.
    # Scenario: convert a temporary .json path used as both input and output. Expects an IO error
    # whose message says "No such file or directory".
    converter = upgrade_schema_lib.Converter()
    _, non_existent = tempfile.mkstemp(suffix=".json")  # safe to ignore fd
    with self.assertRaisesRegex(IOError, "No such file or directory"):
      converter.Convert(non_existent, non_existent)

  def testInvalidExtension(self):
    # [FUNCTION DOC]
    # Scenario: a temporary file with an unsupported extension is used as input, expecting a value
    # error saying "Invalid extension on input". Then a valid empty version 1 JSON input is
    # converted to an output path with the unsupported extension, expecting a value error saying
    # "Invalid extension on output".
    converter = upgrade_schema_lib.Converter()
    _, invalid_extension = tempfile.mkstemp(suffix=".foo")  # safe to ignore fd
    with self.assertRaisesRegex(ValueError, "Invalid extension on input"):
      converter.Convert(invalid_extension, invalid_extension)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_json:
      JsonDumpAndFlush(EMPTY_TEST_SCHEMA_V1, in_json)
      with self.assertRaisesRegex(ValueError, "Invalid extension on output"):
        converter.Convert(in_json.name, invalid_extension)

  def CheckConversion(self, data_old, data_expected):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for the upgrade tests.
    # Writes data_old (a model dictionary of any version) to a temporary JSON file, converts it to
    # JSON, and also to .tflite and then .bin, asserting the .bin and .tflite bytes are identical.
    # Finally asserts that the converted JSON equals data_expected (the upgraded dictionary).
    """Given a data dictionary, test upgrading to current version.

    Args:
        data_old: TFLite model as a dictionary (arbitrary version).
        data_expected: TFLite model as a dictionary (upgraded).
    """
    converter = upgrade_schema_lib.Converter()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_json, \
            tempfile.NamedTemporaryFile(
                suffix=".json", mode="w+") as out_json, \
            tempfile.NamedTemporaryFile(
                suffix=".bin", mode="w+b") as out_bin, \
            tempfile.NamedTemporaryFile(
                suffix=".tflite", mode="w+b") as out_tflite:
      JsonDumpAndFlush(data_old, in_json)
      # Test JSON output
      converter.Convert(in_json.name, out_json.name)
      # Test binary output
      # Convert to .tflite  and then to .bin and check if binary is equal
      converter.Convert(in_json.name, out_tflite.name)
      converter.Convert(out_tflite.name, out_bin.name)
      self.assertEqual(
          open(out_bin.name, "rb").read(),
          open(out_tflite.name, "rb").read())
      # Test that conversion actually produced successful new json.
      converted_schema = json.load(out_json)
      self.assertEqual(converted_schema, data_expected)

  def testAlreadyUpgraded(self):
    # [FUNCTION DOC]
    # Scenario: version 3 models (empty, simple and buffer cases) are converted and must come out
    # unchanged.
    """A file already at version 3 should stay at version 3."""
    self.CheckConversion(EMPTY_TEST_SCHEMA_V3, EMPTY_TEST_SCHEMA_V3)
    self.CheckConversion(TEST_SCHEMA_V3, TEST_SCHEMA_V3)
    self.CheckConversion(BUFFER_TEST_V3, BUFFER_TEST_V3)

  # Disable this while we have incorrectly versioned structures around.
  # def testV0Upgrade_IntroducesSubgraphs(self):
  #   """V0 did not have subgraphs; check to make sure they get introduced."""
  #   self.CheckConversion(TEST_SCHEMA_V0, TEST_SCHEMA_V3)

  def testV1Upgrade_RenameOps(self):
    # [FUNCTION DOC]
    # Scenario: an empty version 1 model and a full version 1 model with old operator code and
    # option type names are converted; expects the empty and fully renamed version 3 models.
    """V1 had many different names for ops; check to make sure they rename."""
    self.CheckConversion(EMPTY_TEST_SCHEMA_V1, EMPTY_TEST_SCHEMA_V3)
    self.CheckConversion(FULL_TEST_SCHEMA_V1, FULL_TEST_SCHEMA_V3)

  def testV2Upgrade_CreateBuffers(self):
    # [FUNCTION DOC]
    # Scenario: a version 2 model with inline tensor data of various sizes (including empty) is
    # converted; expects a version 3 model with a buffers table (empty buffer first) and tensors
    # referring to buffers by index.
    """V2 did not have buffers; check to make sure they are created."""
    self.CheckConversion(BUFFER_TEST_V2, BUFFER_TEST_V3)


if __name__ == "__main__":
  test_lib.main()
