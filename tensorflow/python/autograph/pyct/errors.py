# [MODULE DOC]
# Defines the exception hierarchy used by the code transformation (pyct) library: a base error
# and two specializations for unsupported language elements and inaccessible source code.
# No import-time side effects.
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
"""Code transformation exceptions."""


class PyCTError(Exception):
  # [CLASS DOC]
  # Base class for all errors raised by the code transformation library; carries no extra state.
  """Base class for all exceptions."""


class UnsupportedLanguageElementError(PyCTError, NotImplementedError):
  # [CLASS DOC]
  # Raised for code patterns that autograph cannot convert. Both a library error and a
  # standard not-implemented error, so either can be caught.
  """Raised for code patterns that AutoGraph does not support."""


class InaccessibleSourceCodeError(PyCTError, ValueError):
  # [CLASS DOC]
  # Raised when source code of an entity cannot be obtained through introspection. Both a
  # library error and a standard value error.
  """Raised when inspect can not access source code."""
