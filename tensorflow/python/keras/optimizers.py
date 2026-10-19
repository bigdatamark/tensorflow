# [MODULE DOC]
# Public entry point for Keras built-in optimizers. Re-exports the legacy Optimizer and
# TFOptimizer classes from the v1 optimizer module and provides helper functions to serialize
# an optimizer to a configuration mapping, deserialize one back (case-insensitive for built-in
# names), and resolve a flexible identifier (instance, legacy TF optimizer, mapping or name)
# into an optimizer instance. No import-time side effects beyond imports.
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
# pylint: disable=invalid-name
"""Built-in optimizer classes.

For more examples see the base class `tf.keras.optimizers.Optimizer`.
"""

from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizer_v1 import Optimizer
from tensorflow.python.keras.optimizer_v1 import TFOptimizer
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import ftrl
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.training import optimizer as tf_optimizer_module


def serialize(optimizer):
  # [FUNCTION DOC]
  # Converts an optimizer instance into a JSON-compatible mapping holding its class name and
  # configuration, by delegating to the generic Keras object serializer. Input is a single
  # optimizer instance; output is the mapping. No side effects.
  """Serialize the optimizer configuration to JSON compatible python dict.

  The configuration can be used for persistence and reconstruct the `Optimizer`
  instance again.

  >>> tf.keras.optimizers.serialize(tf.keras.optimizers.SGD())
  {'class_name': 'SGD', 'config': {'name': 'SGD', 'learning_rate': 0.01,
                                   'decay': 0.0, 'momentum': 0.0,
                                   'nesterov': False}}

  Args:
    optimizer: An `Optimizer` instance to serialize.

  Returns:
    Python dict which contains the configuration of the input optimizer.
  """
  return serialize_keras_object(optimizer)


def deserialize(config, custom_objects=None):
  # [FUNCTION DOC]
  # Rebuilds an optimizer from a configuration mapping holding a class name and its settings.
  # The optional custom_objects mapping (names to classes or functions) is consulted in
  # addition to the built-in optimizers (adadelta, adagrad, adam, adamax, nadam, rmsprop, sgd,
  # ftrl and the loss scale optimizer, including its legacy v1 name). The class name is matched
  # case-insensitively for built-ins.
  # KNOWN ISSUE: the config mapping passed in is mutated in place, its class name being
  # replaced by the lower-cased name when it matches a built-in.
  # Returns the new optimizer instance. Errors for unknown names come from the generic Keras
  # deserializer. Imports the loss scale optimizer lazily to avoid a circular dependency.
  """Inverse of the `serialize` function.

  Args:
      config: Optimizer configuration dictionary.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.

  Returns:
      A Keras Optimizer instance.
  """
  # loss_scale_optimizer has a direct dependency of optimizer, import here
  # rather than top to avoid the cyclic dependency.
  from tensorflow.python.keras.mixed_precision import loss_scale_optimizer  # pylint: disable=g-import-not-at-top
  all_classes = {
      'adadelta': adadelta_v2.Adadelta,
      'adagrad': adagrad_v2.Adagrad,
      'adam': adam_v2.Adam,
      'adamax': adamax_v2.Adamax,
      'nadam': nadam_v2.Nadam,
      'rmsprop': rmsprop_v2.RMSprop,
      'sgd': gradient_descent_v2.SGD,
      'ftrl': ftrl.Ftrl,
      'lossscaleoptimizer': loss_scale_optimizer.LossScaleOptimizer,
      # LossScaleOptimizerV1 deserializes into LossScaleOptimizer, as
      # LossScaleOptimizerV1 will be removed soon but deserializing it will
      # still be supported.
      'lossscaleoptimizerv1': loss_scale_optimizer.LossScaleOptimizer,
  }

  # Make deserialization case-insensitive for built-in optimizers.
  if config['class_name'].lower() in all_classes:
    config['class_name'] = config['class_name'].lower()
  return deserialize_keras_object(
      config,
      module_objects=all_classes,
      custom_objects=custom_objects,
      printable_module_name='optimizer')


def get(identifier):
  # [FUNCTION DOC]
  # Resolves an optimizer identifier into a usable optimizer object. Accepts an existing legacy
  # (v1 Keras) or v2 optimizer instance, which is returned unchanged; a legacy TensorFlow
  # training optimizer, which is wrapped in the Keras wrapper class and registered with the
  # backend so its state is tracked; a configuration mapping, which is deserialized; or a
  # string name, which is turned into a configuration with empty settings and deserialized.
  # Raises a value error if the identifier is none of these types. Side effect: registers
  # wrapped legacy optimizers with the backend.
  """Retrieves a Keras Optimizer instance.

  Args:
      identifier: Optimizer identifier, one of
          - String: name of an optimizer
          - Dictionary: configuration dictionary. - Keras Optimizer instance (it
            will be returned unchanged). - TensorFlow Optimizer instance (it
            will be wrapped as a Keras Optimizer).

  Returns:
      A Keras Optimizer instance.

  Raises:
      ValueError: If `identifier` cannot be interpreted.
  """
  if isinstance(identifier, (Optimizer, optimizer_v2.OptimizerV2)):
    return identifier
  # Wrap legacy TF optimizer instances
  elif isinstance(identifier, tf_optimizer_module.Optimizer):
    opt = TFOptimizer(identifier)
    backend.track_tf_optimizer(opt)
    return opt
  elif isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  else:
    raise ValueError(
        'Could not interpret optimizer identifier: {}'.format(identifier))
