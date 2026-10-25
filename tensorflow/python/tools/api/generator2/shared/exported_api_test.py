# [MODULE DOC]
# Unit test for the exported API data model. Defines a module-level fixture collection with one
# module docstring and two symbol records for the same source file, then verifies that writing it
# to a JSON file and reading that file back into a fresh collection gives an equal collection.
# Run through the TensorFlow test runner. No import-time side effects other than the fixture.
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
from tensorflow.python.platform import test
from tensorflow.python.tools.api.generator2.shared import exported_api

_EXPORTS = exported_api.ExportedApi(
    docs=[
        exported_api.ExportedDoc(
            file_name="tf/python/framework/tensor.py",
            line_no=0,
            modules=("tf",),
            docstring="This is a docstring",
        ),
    ],
    symbols=[
        exported_api.ExportedSymbol(
            file_name="tf/python/framework/tensor.py",
            line_no=139,
            symbol_name="Tensor",
            v1_apis=("tf.Tensor",),
            v2_apis=(
                "tf.Tensor",
                "tf.experimental.numpy.ndarray",
            ),
        ),
        exported_api.ExportedSymbol(
            file_name="tf/python/framework/tensor.py",
            line_no=770,
            symbol_name="Tensor",
            v1_apis=("tf.enable_tensor_equality",),
            v2_apis=(),
        ),
    ],
)


class ExportedApiTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the exported API container; uses the module-level fixture collection and the
  # framework-provided temporary directory.

  def test_read_write(self):
    # [FUNCTION DOC]
    # Round-trip test. Setup: write the fixture collection to a JSON file in the temp directory.
    # Action: read that file into a new empty collection. Asserts the new collection equals the
    # fixture, proving write and read are inverse operations for docs and symbols.
    filename = self.get_temp_dir() + "/test_write.json"
    _EXPORTS.write(filename)
    e = exported_api.ExportedApi()
    e.read(filename)

    self.assertEqual(e, _EXPORTS)


if __name__ == "__main__":
  test.main()
