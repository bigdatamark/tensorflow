# [PACKAGE DOC]
# Source-tree placeholder for the top-level tensorflow package. The real build replaces this file
# with one generated from the API template, so this version only exists to let tests run from the
# source tree. Import-time effects: loads the native wrapper module, imports the platform flags
# and app modules and attaches the flags module as an attribute of the app module, then deletes
# the names for the python and core subpackages from the package namespace so they are hidden.
# Exports nothing explicitly.
# KNOWN ISSUE: deleting the python and core names raises a name error if those subpackages have
# not been bound in the namespace (unlike the templates, which guard the deletes).
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

# Bring in all of the public TensorFlow interface into this
# module.

# pylint: disable=g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
from tensorflow.python.platform import app  # pylint: disable=g-import-not-at-top
app.flags = flags

# These symbols appear because we import the python package which
# in turn imports from tensorflow.core and tensorflow.python. They
# must come from this module. So python adds these symbols for the
# resolution to succeed.
# pylint: disable=undefined-variable
del python
del core
# pylint: enable=undefined-variable
