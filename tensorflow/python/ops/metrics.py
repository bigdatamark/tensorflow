# [MODULE DOC]
# HELPER: thin re-export shim that exposes the evaluation metric implementations under a public module name.
# Purpose: provides the public module for evaluation-related metrics (streaming accuracy, precision,
# recall, AUC, mean, and similar). It contains no logic of its own.
# Exposes: every public name defined in the metrics implementation module, via a wildcard import.
# Relation to the package: the real code lives in the metrics implementation module in this same
# ops package; this file only forwards those names so callers can import them from here.
# Import-time side effects: importing this module imports the implementation module (and thereby
# registers whatever that module registers, such as local variables and collections used by metrics).
# Lint warnings for the wildcard import are disabled around the import line.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Evaluation-related metrics."""
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.metrics_impl import *
# pylint: enable=wildcard-import
