# [MODULE DOC]
# HELPER: small helper module for tests of registered objects. Exposes accessors that read the
# private global registries of the registration module (serializable classes and checkpoint
# savers) so tests can inspect everything registered so far. No import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for testing registered objects."""
from tensorflow.python.saved_model.registration import registration as registration_lib
# pylint: disable=protected-access


def get_all_registered_serializables():
  # [FUNCTION DOC]
  # HELPER: test accessor. Returns the live mapping from registered class names to their
  # predicates held by the private global class registry (not a copy). Takes no inputs.
  return registration_lib._class_registry.get_registrations()


def get_all_registered_checkpoint_savers():
  # [FUNCTION DOC]
  # HELPER: test accessor. Returns the live mapping from registered saver names to their
  # predicates held by the private global saver registry (not a copy). Takes no inputs.
  return registration_lib._saver_registry.get_registrations()
