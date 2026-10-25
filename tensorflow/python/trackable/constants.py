# [MODULE DOC]
# HELPER: shared constants for the trackable, checkpoint and saved-model serialization code.
# Defines string keys used inside checkpoints (the object graph proto key, the variable value key
# and the object config JSON key) and the SaveType enumeration. Has no import-time side effects
# beyond defining these names; the only dependency is the standard enum module.

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
"""Constants used in Trackable for checkpointing and serialization."""

import enum


# Key where the object graph proto is saved in a TensorBundle
OBJECT_GRAPH_PROTO_KEY = "_CHECKPOINTABLE_OBJECT_GRAPH"

# A key indicating a variable's value in an object's checkpointed Tensors
# (Trackable._gather_saveables_for_checkpoint). If this is the only key and
# the object has no dependencies, then its value may be restored on object
# creation (avoiding double assignment when executing eagerly).
VARIABLE_VALUE_KEY = "VARIABLE_VALUE"
OBJECT_CONFIG_JSON_KEY = "OBJECT_CONFIG_JSON"


@enum.unique
class SaveType(str, enum.Enum):
  # [CLASS DOC]
  # Enumeration of the two serialization targets a Trackable can be asked to save for. It is both a
  # string and an enum, so members compare equal to their lowercase string values. Members are
  # SAVEDMODEL (value "savedmodel") and CHECKPOINT (value "checkpoint"); the unique decorator
  # forbids aliasing of values. Used when objects choose their saveable behavior per save type.
  SAVEDMODEL = "savedmodel"
  CHECKPOINT = "checkpoint"
