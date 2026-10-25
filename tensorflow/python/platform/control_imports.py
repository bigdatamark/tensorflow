# [MODULE DOC]
# Helper module of boolean switches choosing between Google-internal and open-source
# dependencies. One overall switch (set to open source) and per-dependency switches for app,
# flags, file access, googletest, logging and parameterized testing, all currently true.
# KNOWN ISSUE: the TODO says the per-dependency switches should be flipped and removed, so they
# are transitional; nothing in this file uses them. No functions; import has no side effects.

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
"""Switch between Google or open source dependencies."""
# Switch between Google and OSS dependencies
USE_OSS = True

# Per-dependency switches determining whether each dependency is ready
# to be replaced by its OSS equivalence.
# TODO(danmane,mrry,opensource): Flip these switches, then remove them
OSS_APP = True
OSS_FLAGS = True
OSS_GFILE = True
OSS_GOOGLETEST = True
OSS_LOGGING = True
OSS_PARAMETERIZED = True
