# [MODULE DOC]
# Unit tests for the TFLite model visualizer module (visualize). Covers enum-to-name helpers,
# flatbuffer-to-dictionary conversion and full HTML generation using a mock flatbuffer model
# built by the shared test_utils helper. Run as a test binary; no import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite Python Interface: Sanity check."""
import os
import re

from tensorflow.lite.tools import test_utils
from tensorflow.lite.tools import visualize
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class VisualizeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the public helpers of the visualize module. Has no shared fixtures; each
  # test builds its own mock model or uses plain enum codes.

  def testTensorTypeToName(self):
    # [FUNCTION DOC]
    # Checks that tensor type code zero is converted to the name for 32-bit float.
    self.assertEqual('FLOAT32', visualize.TensorTypeToName(0))

  def testBuiltinCodeToName(self):
    # [FUNCTION DOC]
    # Checks that builtin operator code ten is converted to the hashtable lookup operator name.
    self.assertEqual('HASHTABLE_LOOKUP', visualize.BuiltinCodeToName(10))

  def testFlatbufferToDict(self):
    # [FUNCTION DOC]
    # Builds a mock flatbuffer model, converts it to a dictionary and asserts: the schema version
    # matches the expected constant, there is one subgraph, two operator codes, three buffers,
    # three tensors in the first subgraph, and the first tensor refers to buffer zero.
    model = test_utils.build_mock_flatbuffer_model()
    model_dict = visualize.CreateDictFromFlatbuffer(model)
    self.assertEqual(test_utils.TFLITE_SCHEMA_VERSION, model_dict['version'])
    self.assertEqual(1, len(model_dict['subgraphs']))
    self.assertEqual(2, len(model_dict['operator_codes']))
    self.assertEqual(3, len(model_dict['buffers']))
    self.assertEqual(3, len(model_dict['subgraphs'][0]['tensors']))
    self.assertEqual(0, model_dict['subgraphs'][0]['tensors'][0]['buffer'])

  def testVisualize(self):
    # [FUNCTION DOC]
    # Writes a mock flatbuffer model to a file in the temporary directory, generates the HTML
    # for that file path, and asserts the output mentions the file name, the input and constant
    # tensor names and the add operator. Only a sanity check, not a full HTML parse.
    model = test_utils.build_mock_flatbuffer_model()
    tmp_dir = self.get_temp_dir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')
    with open(model_filename, 'wb') as model_file:
      model_file.write(model)

    html_text = visualize.create_html(model_filename)

    # It's hard to test debug output without doing a full HTML parse,
    # but at least sanity check that expected identifiers are present.
    self.assertRegex(
        html_text, re.compile(r'%s' % model_filename, re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text,
                     re.compile(r'input_tensor', re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text,
                     re.compile(r'constant_tensor', re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text, re.compile(r'ADD', re.MULTILINE | re.DOTALL))


if __name__ == '__main__':
  test.main()
