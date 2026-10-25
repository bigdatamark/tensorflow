# [MODULE DOC]
# Test module for the lazy loader: checks that documentation-test discovery does not load the
# module, that the warning is emitted once on first load, and that pickling yields the real module.
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
"""lazy loader tests."""

# pylint: disable=unused-import
import doctest
import inspect
import pickle
import types

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import tf_inspect


class LazyLoaderTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the lazy loader: documentation-test discovery must not trigger a load, and a
  # configured warning is emitted exactly once on first load.

  def testDocTestDoesNotLoad(self):
    # [FUNCTION DOC]
    # Scenario: a module object holding a lazy loader for a standard path module; run the
    # documentation-test finder over it.
    # Asserts the attribute is a lazy loader both before and after (i.e. nothing was loaded).
    module = types.ModuleType("mytestmodule")
    module.foo = lazy_loader.LazyLoader("foo", module.__dict__, "os.path")

    self.assertIsInstance(module.foo, lazy_loader.LazyLoader)

    finder = doctest.DocTestFinder()
    finder.find(module)

    self.assertIsInstance(module.foo, lazy_loader.LazyLoader)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def testLazyLoaderMock(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: the warning logging function is replaced by a mock (passed in as mock_warning);
    # a lazy loader targeting this very test module is created with a warning message.
    # Asserts: no warning at creation; setting an attribute loads the module and logs exactly
    # one warning; reading it back logs no further warning; the value is unchanged.
    name = LazyLoaderTest.__module__
    lazy_loader_module = lazy_loader.LazyLoader(
        "lazy_loader_module", globals(), name, warning="Test warning.")

    self.assertEqual(0, mock_warning.call_count)
    lazy_loader_module.foo = 0
    self.assertEqual(1, mock_warning.call_count)
    foo = lazy_loader_module.foo
    self.assertEqual(1, mock_warning.call_count)

    # Check that values stayed the same
    self.assertEqual(lazy_loader_module.foo, foo)


class PickleTest(test.TestCase):
  # [CLASS DOC]
  # Tests pickling support of the lazy loader.

  def testPickleLazyLoader(self):
    # [FUNCTION DOC]
    # Scenario: lazy loader pointing at the current test module is pickled and unpickled.
    # Asserts the restored object is the real module (name matches, and it exposes this file's
    # PickleTest class).
    name = PickleTest.__module__  # Try to pickle current module.
    lazy_loader_module = lazy_loader.LazyLoader(
        "lazy_loader_module", globals(), name)
    restored = pickle.loads(pickle.dumps(lazy_loader_module))
    self.assertEqual(restored.__name__, name)
    self.assertIsNotNone(restored.PickleTest)


if __name__ == "__main__":
  test.main()
