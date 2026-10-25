# [MODULE DOC]
# HELPER: convenience re-export module for the saved_model package.
# Aggregates the public saved-model API in a single namespace: the builder, constants, loader,
# main_op, method_name_updater, signature_constants, signature_def_utils, tag_constants and utils
# submodules, plus the Fingerprint class, read_fingerprint, load, save, and (via wildcard import)
# the simple_save helpers. It defines no functions or classes itself; importing it only imports
# those modules (their import-time registrations apply). The names are re-exported unchanged.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Convenience functions to save a model.
"""


# pylint: disable=unused-import
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import main_op
from tensorflow.python.saved_model import method_name_updater
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model.fingerprinting import Fingerprint
from tensorflow.python.saved_model.fingerprinting import read_fingerprint
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
# pylint: enable=unused-import
# pylint: disable=wildcard-import
from tensorflow.python.saved_model.simple_save import *
# pylint: enable=wildcard-import
