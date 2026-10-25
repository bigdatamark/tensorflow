# [MODULE DOC]
# Tests that saving a module with the experimental image format option writes a chunked proto
# file when the model exceeds the splitter size limit and an ordinary proto file otherwise. Uses
# the splitter's global debug maximum size setting, which the tests change and do not restore.
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
"""Tests for SavedModel save using experimental_image_format."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.module import module
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.tools.proto_splitter import constants


class ProtoSplitterSaveTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for saving a module with the experimental image format option, which writes the
  # saved model as chunked proto when it is bigger than the splitter's maximum size. Tests set
  # the splitter's debug maximum size to force each outcome.

  def test_save_experimental_image_format(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: sets the global splitter maximum size to 80000 bytes and never restores it, so
    # later tests in the process see this limit.
    # Scenario: a module holding a large random float constant (150 by 150) and a function
    # returning it is saved with the image format option and a small size limit. Asserts the
    # chunked proto file (.cpb) exists in the save directory.
    root = module.Module()
    root.c = constant_op.constant(np.random.random_sample([150, 150]))
    root.get_c = def_function.function(lambda: root.c)
    save_dir = os.path.join(self.get_temp_dir(), "chunked_model")
    constants.debug_set_max_size(80000)
    options = save_options.SaveOptions(experimental_image_format=True)
    save.save(
        root,
        save_dir,
        signatures=root.get_c.get_concrete_function(),
        options=options,
    )
    self.assertTrue(os.path.exists(save_dir + "/saved_model.cpb"))

  def test_save_experimental_image_format_not_chunked(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: sets the global splitter maximum size to 2 GB and never restores it.
    # Scenario: the same module is saved with the image format option but a very large size
    # limit, so no splitting is needed. Asserts the ordinary binary proto file (.pb) exists.
    root = module.Module()
    root.c = constant_op.constant(np.random.random_sample([150, 150]))
    root.get_c = def_function.function(lambda: root.c)
    save_dir = os.path.join(self.get_temp_dir(), "not_chunked_model")
    constants.debug_set_max_size(1 << 31)  # 2GB
    options = save_options.SaveOptions(experimental_image_format=True)
    save.save(
        root,
        save_dir,
        signatures=root.get_c.get_concrete_function(),
        options=options,
    )
    # Should save an unchunked proto (.pb) and not .cpb
    self.assertTrue(os.path.exists(save_dir + "/saved_model.pb"))


if __name__ == "__main__":
  test.main()
