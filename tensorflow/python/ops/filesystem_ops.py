# [MODULE DOC]
# Filesystem related Python operations. Currently exposes one wrapper that sets a configuration
# option of a registered file system by calling the generated filesystem op. Thin wrapper
# over the generated ops module; no import-time side effects beyond imports.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Filesystem related operations."""

from tensorflow.python.ops import gen_filesystem_ops as _gen_filesystem_ops


# pylint: disable=protected-access
def filesystem_set_configuration(scheme, key, value, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns None, but the code returns whatever the generated
  # op returns (an operation in graph mode).
  # Sets one configuration option on the file system registered for a URI scheme.
  # Inputs: scheme, the file system scheme (string tensor or value); key, the option name;
  # value, the option value; name, optional operation name.
  # Output: the result of the generated op.
  # Side effect: changes the runtime configuration of the file system when the op executes.
  """Set configuration of the file system.

  Args:
    scheme: File system scheme.
    key: The name of the configuration option.
    value: The value of the configuration option.
    name: A name for the operation (optional).

  Returns:
    None.
  """

  return _gen_filesystem_ops.file_system_set_configuration(
      scheme, key=key, value=value, name=name)


# pylint: enable=protected-access
