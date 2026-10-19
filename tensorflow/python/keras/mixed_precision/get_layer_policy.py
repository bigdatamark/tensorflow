# [MODULE DOC]
# Defines the single function get_layer_policy, which returns the dtype policy of a Keras layer.
# It lives in its own module to avoid a circular import: it needs the base layer module, which
# itself depends on the policy module. No import-time side effects.
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
"""Contains the get_layer_policy function.

This is a separate file from policy.py to avoid a circular dependency.
get_layer_policy() relies on base_layer.py, itself which relies on policy.py.
"""

from tensorflow.python.keras.engine import base_layer


def get_layer_policy(layer):
  # [FUNCTION DOC]
  # Returns the dtype policy attached to a layer. Input: layer, which must be an instance of the
  # Keras base layer class. Output: the layer's dtype policy object. Raises a value error if the
  # argument is not a layer. No side effects.
  """Returns the dtype policy of a layer.

  Warning: This function is deprecated. Use
  `tf.keras.layers.Layer.dtype_policy` instead.

  Args:
    layer: A `tf.keras.layers.Layer`.

  Returns:
    The `tf.keras.mixed_precision.Policy` of the layer.
  """
  if not isinstance(layer, base_layer.Layer):
    raise ValueError('get_policy can only be called on a layer, but got: %s'
                     % (layer,))
  return layer.dtype_policy
