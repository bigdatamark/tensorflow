# [MODULE DOC]
# Tests for the resource loader: reading a nonexistent resource must fail with an IO error, and
# reading a known file in the platform package must return content mentioning tensorflow. Uses
# the googletest router; runs its main when executed as a script.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.platform import googletest
from tensorflow.python.platform import resource_loader


class ResourceLoaderTest(googletest.TestCase):
  # [CLASS DOC]
  # Test case for the resource loading function; no shared fixtures.

  def test_exception(self):
    # [FUNCTION DOC]
    # Loads a resource at a path that does not exist and asserts an IO error is raised.
    with self.assertRaises(IOError):
      resource_loader.load_resource("/fake/file/path/dne")

  def test_exists(self):
    # [FUNCTION DOC]
    # Loads the resource loader's own source file via its path relative to the tensorflow
    # directory and asserts the returned bytes contain the word tensorflow.
    contents = resource_loader.load_resource(
        "python/platform/resource_loader.py")
    self.assertIn(b"tensorflow", contents)


if __name__ == "__main__":
  googletest.main()
