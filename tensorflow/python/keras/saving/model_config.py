# [MODULE DOC]
# Helpers that rebuild Keras models from serialized configurations: from a configuration mapping
# and from a JSON string (a removed YAML entry point remains and only raises an error). They
# delegate to the Keras layer deserializer, imported lazily to avoid import cycles. Used by the
# HDF5 loading code and re-exported as public model utilities. No import-time side effects.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Functions that save the model's config into different formats."""

from tensorflow.python.keras.saving.saved_model import json_utils


def model_from_config(config, custom_objects=None):
  # [FUNCTION DOC]
  # Creates an uncompiled model from a configuration. Inputs: config (a mapping describing the
  # model; a list is rejected) and custom_objects (optional mapping from names to custom classes
  # or functions for deserialization). Raises a type error if config is a list (suggesting the
  # sequential model's own from-config method). Otherwise delegates to the layer deserializer
  # (imported lazily) and returns the model it builds.
  """Instantiates a Keras model from its config.

  Usage:
  ```
  # for a Functional API model
  tf.keras.Model().from_config(model.get_config())

  # for a Sequential model
  tf.keras.Sequential().from_config(model.get_config())
  ```

  Args:
      config: Configuration dictionary.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).

  Raises:
      TypeError: if `config` is not a dictionary.
  """
  if isinstance(config, list):
    raise TypeError('`model_from_config` expects a dictionary, not a list. '
                    'Maybe you meant to use '
                    '`Sequential.from_config(config)`?')
  from tensorflow.python.keras.layers import deserialize  # pylint: disable=g-import-not-at-top
  return deserialize(config, custom_objects=custom_objects)


def model_from_yaml(yaml_string, custom_objects=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring describes parsing YAML and returning a model, but the function was
  # removed for security reasons: both inputs are ignored and it always raises a runtime error.
  # Inputs (yaml_string, custom_objects) are unused. Always raises a runtime error telling the
  # caller to use the JSON serialization methods instead. Never returns.
  """Parses a yaml model configuration file and returns a model instance.

  Note: Since TF 2.6, this method is no longer supported and will raise a
  RuntimeError.

  Args:
      yaml_string: YAML string or open file encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).

  Raises:
      RuntimeError: announces that the method poses a security risk
  """
  raise RuntimeError(
      'Method `model_from_yaml()` has been removed due to security risk of '
      'arbitrary code execution. Please use `Model.to_json()` and '
      '`model_from_json()` instead.'
  )


def model_from_json(json_string, custom_objects=None):
  # [FUNCTION DOC]
  # Creates an uncompiled model from a JSON string. Inputs: json_string (JSON text of a model
  # configuration) and custom_objects (optional mapping from names to custom classes/functions).
  # Decodes the text with the Keras JSON decoder (restoring special encoded types), then
  # delegates to the lazily imported layer deserializer. Returns the model; malformed JSON or
  # unknown classes raise the underlying decoding or value errors.
  """Parses a JSON model configuration string and returns a model instance.

  Usage:

  >>> model = tf.keras.Sequential([
  ...     tf.keras.layers.Dense(5, input_shape=(3,)),
  ...     tf.keras.layers.Softmax()])
  >>> config = model.to_json()
  >>> loaded_model = tf.keras.models.model_from_json(config)

  Args:
      json_string: JSON string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).
  """
  config = json_utils.decode(json_string)
  from tensorflow.python.keras.layers import deserialize  # pylint: disable=g-import-not-at-top
  return deserialize(config, custom_objects=custom_objects)
