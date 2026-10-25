# [MODULE DOC]
# HELPER: Fixture module for the traverse tests, imported by the first fixture module. Defines
# one empty class so traversal can verify that classes in nested modules are reached. No logic
# and no import-time side effects.
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


class ModuleClass2(object):
  # [CLASS DOC]
  # HELPER: Empty fixture class for the traverse tests; has a no-op constructor and a no-op
  # dunder-style method, and holds no state.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: No-op constructor for the fixture. No inputs, returns nothing, no side effects.
    pass

  def __model_class1_method__(self):
    # [FUNCTION DOC]
    # HELPER: No-op placeholder method returning nothing. KNOWN ISSUE: the name refers to
    # class 1 and "model", although it lives in the second module's class; it looks copied from
    # the first fixture module. Harmless.
    pass

