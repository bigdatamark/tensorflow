# [MODULE DOC]
# HELPER: small helper module for code that reads TFLite flatbuffer operator codes.
# Exposes one function that extracts the builtin operator code from a flatbuffer operator-code
# object, hiding the split between the legacy (deprecated, byte-sized) field and the newer
# extended builtin code field. Calls a utility at import time that removes every public symbol
# of this module not on the allowed list, so only the one function stays public.
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
"""Schema utilities to get builtin code from operator code."""

from tensorflow.python.util import all_util


def get_builtin_code_from_operator_code(opcode):
  # [FUNCTION DOC]
  # HELPER: lets converters/tests read operator codes regardless of flatbuffer API flavor.
  # Takes an operator-code object, either a generated reader object (with accessor methods) or
  # an unpacked object-API object (with plain attributes). If a callable builtin-code accessor
  # exists, returns the larger of the builtin code and the deprecated builtin code obtained by
  # calling accessors; otherwise returns the larger of the two corresponding attributes.
  # Taking the maximum works because the legacy field saturates at a placeholder value while the
  # extended field holds the real code. Returns an integer; raises an attribute error if the
  # object has neither form. No side effects.
  """Return the builtin code of the given operator code.

  The following method is introduced to resolve op builtin code shortage
  problem. The new builtin operator will be assigned to the extended builtin
  code field in the flatbuffer schema. Those methods helps to hide builtin code
  details.

  Args:
    opcode: Operator code.

  Returns:
    The builtin code of the given operator code.
  """
  # Access BuiltinCode() method first if available.
  if hasattr(opcode, 'BuiltinCode') and callable(opcode.BuiltinCode):
    return max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())

  return max(opcode.builtinCode, opcode.deprecatedBuiltinCode)


_allowed_symbols = [
    'get_builtin_code_from_operator_code',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
