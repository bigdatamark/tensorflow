# [MODULE DOC]
# HELPER: open-source defaults for platform-specific remote-server helpers. Provides small
# functions answering which communication protocol to use, whether a path is remote, which
# encoding to use for appendable files, and which coordination service type to use. Internal
# builds presumably replace this module. No import-time side effects.

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Platform-specific helpers for connecting to remote servers."""


def get_default_communication_protocol():
  # [FUNCTION DOC]
  # HELPER: platform hook for remote cluster setup.
  # Returns the string naming the default communication protocol for remote servers: grpc.
  return 'grpc'


def is_remote_path(_):
  # [FUNCTION DOC]
  # HELPER: platform hook for path handling.
  # Takes a path (ignored) and always returns false: in this build no path is treated as remote.
  return False


def get_appendable_file_encoding():
  # [FUNCTION DOC]
  # HELPER: platform hook for file writing.
  # Returns the encoding name to use for appendable files; here an empty string, meaning none.
  return ''


def coordination_service_type(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: platform hook for choosing the distributed coordination service implementation.
  # Accepts and discards any positional and keyword arguments, and always returns the null value,
  # meaning no specific service type is selected.
  del args, kwargs
  return None
