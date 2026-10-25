# [MODULE DOC]
# Defines the public tf.initializers-style namespace for the v1 API. It contains no logic of its
# own: it binds short names (zeros, ones, constant, random_uniform, random_normal,
# truncated_normal, uniform_unit_scaling, variance_scaling, orthogonal, identity) to the v1
# initializer classes in the init_ops module, and binds variables, global_variables and
# local_variables to the variable-initializer op builders from the variables module.
# Import-time side effects: only the alias assignments; after that the module removes its
# references to the init_ops and variables modules (it "seals" the API) so they are not exposed.
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
"""Public API for tf.initializer namespace."""

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables as _variables

# variable initializers
zeros = init_ops.zeros_initializer
ones = init_ops.ones_initializer
constant = init_ops.constant_initializer
random_uniform = init_ops.random_uniform_initializer
random_normal = init_ops.random_normal_initializer
truncated_normal = init_ops.truncated_normal_initializer
uniform_unit_scaling = init_ops.uniform_unit_scaling_initializer
variance_scaling = init_ops.variance_scaling_initializer
orthogonal = init_ops.orthogonal_initializer
identity = init_ops.identity_initializer

# variable initializer ops
variables = _variables.variables_initializer
global_variables = _variables.global_variables_initializer
local_variables = _variables.local_variables_initializer

# Seal API.
del init_ops
del _variables
