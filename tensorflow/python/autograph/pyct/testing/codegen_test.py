# [MODULE DOC]
# Test for the random code generator in the sibling codegen module. It generates many random
# function definitions with a fixed random seed and checks each one can be turned into source
# code and loaded as an object by the compiler utility. No import-time side effects.
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
"""Tests for type_info module."""

import numpy as np

from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct.testing import codegen
from tensorflow.python.platform import test


class CodeGenTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the random function definition generator.

  def test_codegen_gens(self):
    # [FUNCTION DOC]
    # Seeds numpy's random generator with zero, then 1000 times generates a random function
    # definition tree and converts it to a live function object via the compiler utility.
    # Asserts each result is not null, i.e. the generated tree is valid and convertible to source.
    np.random.seed(0)
    for _ in range(1000):
      node = codegen.generate_random_functiondef()
      fn = compiler.ast_to_object(node)
      self.assertIsNotNone(
          fn, 'Generated invalid AST that could not convert to source.')


if __name__ == '__main__':
  test.main()
