# [MODULE DOC]
# HELPER: Import-and-configure shim used by the TF NumPy test files in this directory.
# Provides a single namespace that mimics the public TF NumPy API (array, math, dtype and random
# routines, ndarray, bfloat16, finfo, promote_types, result_type, dtype configuration helpers)
# by star-importing the numpy_ops implementation modules, plus the plain numpy module under
# the name onp. Tests import this module as their "np" so they can run unchanged against
# TF NumPy.
# Import-time side effects: sets the default dtype used by the random-normal generator to
# 32-bit float, enables TensorFlow v2 behavior globally, and enables NumPy-style behavior
# (type promotion and tensor methods) globally. There are no functions or classes here.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""TF NumPy API wrapper for the tests."""

# pylint: disable=wildcard-import
# pylint: disable=unused-import
# pylint: disable=g-importing-member

import numpy as onp
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework.dtypes import bfloat16
from tensorflow.python.ops.numpy_ops import np_random as random
from tensorflow.python.ops.numpy_ops.np_array_ops import *
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray
from tensorflow.python.ops.numpy_ops.np_config import enable_numpy_behavior
from tensorflow.python.ops.numpy_ops.np_dtypes import *
from tensorflow.python.ops.numpy_ops.np_dtypes import canonicalize_dtype
from tensorflow.python.ops.numpy_ops.np_dtypes import default_float_type
from tensorflow.python.ops.numpy_ops.np_dtypes import is_allow_float64
from tensorflow.python.ops.numpy_ops.np_dtypes import set_allow_float64
from tensorflow.python.ops.numpy_ops.np_math_ops import *
from tensorflow.python.ops.numpy_ops.np_utils import finfo
from tensorflow.python.ops.numpy_ops.np_utils import promote_types
from tensorflow.python.ops.numpy_ops.np_utils import result_type

random.DEFAULT_RANDN_DTYPE = onp.float32
# pylint: enable=unused-import

v2_compat.enable_v2_behavior()
# TODO(b/171429739): This should be moved to every individual file/test.
enable_numpy_behavior()
