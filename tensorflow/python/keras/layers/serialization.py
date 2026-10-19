# [MODULE DOC]
# Serialization and deserialization entry points for built-in Keras layers. Exposes the layer
# serialize and deserialize functions, a populate function that fills a per-thread mapping from
# layer names to layer classes (all layer classes from the listed layer modules, V2 wrapper
# classes overriding when TF2 behavior is enabled, plus model classes, the input helpers and
# function-style merge layers), and the constants listing the searched modules. Import-time side
# effects: imports many layer modules and creates a thread-local storage object; the mapping
# itself is built lazily on first deserialize.
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
"""Layer serialization/deserialization functions.
"""
# pylint: disable=wildcard-import
# pylint: disable=unused-import

import threading

from tensorflow.python import tf2
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import convolutional_recurrent
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers import pooling
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import rnn_cell_wrapper_v2
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect as inspect

ALL_MODULES = (base_layer, input_layer, advanced_activations, convolutional,
               convolutional_recurrent, core, dense_attention,
               embeddings, merge, pooling, recurrent)
ALL_V2_MODULES = (rnn_cell_wrapper_v2,)
# ALL_OBJECTS is meant to be a global mutable. Hence we need to make it
# thread-local to avoid concurrent mutations.
LOCAL = threading.local()


def populate_deserializable_objects():
  # [FUNCTION DOC]
  # HELPER: lazily prepares the name-to-class lookup used by deserialize.
  # KNOWN ISSUE: the docstring says it fills a dict named ALL_OBJECTS, but the mapping actually
  # lives as an attribute of the thread-local storage object (the module-level name ALL_OBJECTS
  # does not exist); the global declaration for the storage object is also unnecessary.
  # Takes no arguments and returns nothing. Ensures the current thread has a mapping from names
  # to layer classes. If one already exists and was generated under the same TF2 setting, it
  # returns immediately. Otherwise it rebuilds: adds every Layer subclass found in the listed
  # layer modules, then (only when TF2 is enabled) overrides with the V2 wrapper classes, then
  # lazily imports the models module (avoiding a circular import) and adds the input function,
  # input spec, functional, model and sequential classes plus the function-style merge layers
  # (add, subtract, multiply, average, maximum, minimum, concatenate, dot). Mutates thread-local
  # state only.
  """Populates dict ALL_OBJECTS with every built-in layer.
  """
  global LOCAL
  if not hasattr(LOCAL, 'ALL_OBJECTS'):
    LOCAL.ALL_OBJECTS = {}
    LOCAL.GENERATED_WITH_V2 = None

  if LOCAL.ALL_OBJECTS and LOCAL.GENERATED_WITH_V2 == tf2.enabled():
    # Objects dict is already generated for the proper TF version:
    # do nothing.
    return

  LOCAL.ALL_OBJECTS = {}
  LOCAL.GENERATED_WITH_V2 = tf2.enabled()

  base_cls = base_layer.Layer
  generic_utils.populate_dict_with_module_objects(
      LOCAL.ALL_OBJECTS,
      ALL_MODULES,
      obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls))

  # Overwrite certain V1 objects with V2 versions
  if tf2.enabled():
    generic_utils.populate_dict_with_module_objects(
        LOCAL.ALL_OBJECTS,
        ALL_V2_MODULES,
        obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls))

  # Prevent circular dependencies.
  from tensorflow.python.keras import models  # pylint: disable=g-import-not-at-top

  LOCAL.ALL_OBJECTS['Input'] = input_layer.Input
  LOCAL.ALL_OBJECTS['InputSpec'] = input_spec.InputSpec
  LOCAL.ALL_OBJECTS['Functional'] = models.Functional
  LOCAL.ALL_OBJECTS['Model'] = models.Model
  LOCAL.ALL_OBJECTS['Sequential'] = models.Sequential

  # Merge layers, function versions.
  LOCAL.ALL_OBJECTS['add'] = merge.add
  LOCAL.ALL_OBJECTS['subtract'] = merge.subtract
  LOCAL.ALL_OBJECTS['multiply'] = merge.multiply
  LOCAL.ALL_OBJECTS['average'] = merge.average
  LOCAL.ALL_OBJECTS['maximum'] = merge.maximum
  LOCAL.ALL_OBJECTS['minimum'] = merge.minimum
  LOCAL.ALL_OBJECTS['concatenate'] = merge.concatenate
  LOCAL.ALL_OBJECTS['dot'] = merge.dot


def serialize(layer):
  # [FUNCTION DOC]
  # Converts a layer object into its serializable configuration mapping (class name plus config)
  # by delegating to the generic Keras object serializer. Input: the layer instance. Returns the
  # mapping. Errors from the generic serializer propagate. No side effects.
  return generic_utils.serialize_keras_object(layer)


def deserialize(config, custom_objects=None):
  # [FUNCTION DOC]
  # Rebuilds a layer (or model) from a configuration mapping holding a class name and a config.
  # Inputs: the config mapping; optional mapping of custom class or function names to objects
  # consulted in addition to the built-ins. First ensures the thread's built-in layer mapping is
  # populated, then delegates to the generic Keras deserializer, labelling errors as about a
  # layer. Returns the new instance; unknown class names make the generic deserializer raise a
  # value error. Side effect: may populate the thread-local mapping.
  """Instantiates a layer from a config dictionary.

  Args:
      config: dict of the form {'class_name': str, 'config': dict}
      custom_objects: dict mapping class names (or function names)
          of custom (non-Keras) objects to class/functions

  Returns:
      Layer instance (may be Model, Sequential, Network, Layer...)
  """
  populate_deserializable_objects()
  return generic_utils.deserialize_keras_object(
      config,
      module_objects=LOCAL.ALL_OBJECTS,
      custom_objects=custom_objects,
      printable_module_name='layer')
