# [MODULE DOC]
# HELPER: Fixture module for the traverse tests. Defines one class that holds an instance of the
# class from the sibling fixture module, so a traversal starting here reaches classes in both
# modules. Contains no logic and no import-time side effects beyond importing that sibling.
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
"""A module target for TraverseTest.test_module."""

from tensorflow.tools.common import test_module2


class ModuleClass1(object):
  # [CLASS DOC]
  # HELPER: Fixture class for the traverse tests. Holds a private attribute referencing an
  # instance of the class from the second fixture module and defines a dunder-style method.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: Constructor for the fixture. Takes no inputs; creates an instance of the second
    # module's class and stores it in a private attribute. Returns nothing.
    self._m2 = test_module2.ModuleClass2()

  def __model_class1_method__(self):
    # [FUNCTION DOC]
    # HELPER: Placeholder method that does nothing and returns nothing; exists only so the class
    # has a dunder-named member for traversal tests. KNOWN ISSUE: the name says "model" but is
    # presumably a typo for "module"; it has no effect on behavior.
    pass

