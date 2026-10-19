# [MODULE DOC]
# HELPER: aggregation module that gathers the feature column API into a single import location.
# Re-export facade for the feature column package. It defines no classes or functions of its own;
# it only pulls every public name from the version one feature column module, the version two
# feature column module, the sequence feature column module and the serialization module into
# one namespace, using wildcard imports. Consumers import this module to get the complete set of
# feature column constructors, helper classes and (de)serialization utilities in one place.
# Import-time side effects: importing it imports the four submodules (so any registrations or
# global state they create happen), and name collisions are resolved by import order, with later
# modules overriding earlier ones. Lint checks for unused imports and wildcards are disabled.
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
"""FeatureColumns: tools for ingesting and representing features."""

# pylint: disable=unused-import,line-too-long,wildcard-import,g-bad-import-order
from tensorflow.python.feature_column.feature_column import *
from tensorflow.python.feature_column.feature_column_v2 import *
from tensorflow.python.feature_column.sequence_feature_column import *
from tensorflow.python.feature_column.serialization import *
# pylint: enable=unused-import,line-too-long
