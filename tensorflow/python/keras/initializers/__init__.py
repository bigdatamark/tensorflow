# [PACKAGE DOC]
# Package for Keras weight initializers (legacy copy under tensorflow.python.keras). It does not
# define initializer classes itself; it provides a registry that maps names (both class-style and
# snake-case aliases) to initializer classes, plus the serialize, deserialize and get helpers
# used by layers to turn names, configuration mappings or classes into initializer objects.
# The registry is chosen according to whether TF2 behavior is enabled: the version 2 classes from
# initializers_v2 or the version 1 classes from initializers_v1 and the graph-mode init_ops.
# Import-time side effects: the registry is built once, and every registry entry is also copied
# into this module's global namespace so that names can be accessed as module attributes.
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
"""Keras initializer serialization / deserialization."""

import threading

from tensorflow.python import tf2
from tensorflow.python.keras.initializers import initializers_v1
from tensorflow.python.keras.initializers import initializers_v2
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect as inspect
from tensorflow.python.ops import init_ops


# LOCAL.ALL_OBJECTS is meant to be a global mutable. Hence we need to make it
# thread-local to avoid concurrent mutations.
LOCAL = threading.local()


def populate_deserializable_objects():
  # [FUNCTION DOC]
  # HELPER: builds the name-to-class registry used by deserialize and get in this package.
  # Fills the thread-local registry of built-in initializers. Takes no inputs and returns nothing.
  # If the registry already exists for the current TF2-enabled state it does nothing; otherwise it
  # resets the registry and records which TF version it was generated for. It always adds the
  # version-2 compatibility aliases (V2-suffixed class names and legacy snake-case names). When TF2
  # is enabled it adds every Initializer subclass found in initializers_v2 under its class name and
  # its snake-case name; otherwise it adds a fixed table of version 1 classes under class and
  # snake-case names. Finally it adds short aliases: normal, uniform, one and zero.
  # Side effects: mutates the thread-local registry state.
  """Populates dict ALL_OBJECTS with every built-in initializer.
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

  # Compatibility aliases (need to exist in both V1 and V2).
  LOCAL.ALL_OBJECTS['ConstantV2'] = initializers_v2.Constant
  LOCAL.ALL_OBJECTS['GlorotNormalV2'] = initializers_v2.GlorotNormal
  LOCAL.ALL_OBJECTS['GlorotUniformV2'] = initializers_v2.GlorotUniform
  LOCAL.ALL_OBJECTS['HeNormalV2'] = initializers_v2.HeNormal
  LOCAL.ALL_OBJECTS['HeUniformV2'] = initializers_v2.HeUniform
  LOCAL.ALL_OBJECTS['IdentityV2'] = initializers_v2.Identity
  LOCAL.ALL_OBJECTS['LecunNormalV2'] = initializers_v2.LecunNormal
  LOCAL.ALL_OBJECTS['LecunUniformV2'] = initializers_v2.LecunUniform
  LOCAL.ALL_OBJECTS['OnesV2'] = initializers_v2.Ones
  LOCAL.ALL_OBJECTS['OrthogonalV2'] = initializers_v2.Orthogonal
  LOCAL.ALL_OBJECTS['RandomNormalV2'] = initializers_v2.RandomNormal
  LOCAL.ALL_OBJECTS['RandomUniformV2'] = initializers_v2.RandomUniform
  LOCAL.ALL_OBJECTS['TruncatedNormalV2'] = initializers_v2.TruncatedNormal
  LOCAL.ALL_OBJECTS['VarianceScalingV2'] = initializers_v2.VarianceScaling
  LOCAL.ALL_OBJECTS['ZerosV2'] = initializers_v2.Zeros

  # Out of an abundance of caution we also include these aliases that have
  # a non-zero probability of having been included in saved configs in the past.
  LOCAL.ALL_OBJECTS['glorot_normalV2'] = initializers_v2.GlorotNormal
  LOCAL.ALL_OBJECTS['glorot_uniformV2'] = initializers_v2.GlorotUniform
  LOCAL.ALL_OBJECTS['he_normalV2'] = initializers_v2.HeNormal
  LOCAL.ALL_OBJECTS['he_uniformV2'] = initializers_v2.HeUniform
  LOCAL.ALL_OBJECTS['lecun_normalV2'] = initializers_v2.LecunNormal
  LOCAL.ALL_OBJECTS['lecun_uniformV2'] = initializers_v2.LecunUniform

  if tf2.enabled():
    # For V2, entries are generated automatically based on the content of
    # initializers_v2.py.
    v2_objs = {}
    base_cls = initializers_v2.Initializer
    generic_utils.populate_dict_with_module_objects(
        v2_objs,
        [initializers_v2],
        obj_filter=lambda x: inspect.isclass(x) and issubclass(x, base_cls))
    for key, value in v2_objs.items():
      LOCAL.ALL_OBJECTS[key] = value
      # Functional aliases.
      LOCAL.ALL_OBJECTS[generic_utils.to_snake_case(key)] = value
  else:
    # V1 initializers.
    v1_objs = {
        'Constant': init_ops.Constant,
        'GlorotNormal': init_ops.GlorotNormal,
        'GlorotUniform': init_ops.GlorotUniform,
        'Identity': init_ops.Identity,
        'Ones': init_ops.Ones,
        'Orthogonal': init_ops.Orthogonal,
        'VarianceScaling': init_ops.VarianceScaling,
        'Zeros': init_ops.Zeros,
        'HeNormal': initializers_v1.HeNormal,
        'HeUniform': initializers_v1.HeUniform,
        'LecunNormal': initializers_v1.LecunNormal,
        'LecunUniform': initializers_v1.LecunUniform,
        'RandomNormal': initializers_v1.RandomNormal,
        'RandomUniform': initializers_v1.RandomUniform,
        'TruncatedNormal': initializers_v1.TruncatedNormal,
    }
    for key, value in v1_objs.items():
      LOCAL.ALL_OBJECTS[key] = value
      # Functional aliases.
      LOCAL.ALL_OBJECTS[generic_utils.to_snake_case(key)] = value

  # More compatibility aliases.
  LOCAL.ALL_OBJECTS['normal'] = LOCAL.ALL_OBJECTS['random_normal']
  LOCAL.ALL_OBJECTS['uniform'] = LOCAL.ALL_OBJECTS['random_uniform']
  LOCAL.ALL_OBJECTS['one'] = LOCAL.ALL_OBJECTS['ones']
  LOCAL.ALL_OBJECTS['zero'] = LOCAL.ALL_OBJECTS['zeros']


# For backwards compatibility, we populate this file with the objects
# from ALL_OBJECTS. We make no guarantees as to whether these objects will
# using their correct version.
populate_deserializable_objects()
globals().update(LOCAL.ALL_OBJECTS)

# Utility functions


def serialize(initializer):
  # [FUNCTION DOC]
  # Converts an initializer object into its serializable configuration mapping (class name plus
  # config) by delegating to the generic Keras object serializer. Input: an initializer instance.
  # Returns the mapping. No side effects.
  return generic_utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  # [FUNCTION DOC]
  # Rebuilds an initializer from a name string or a configuration mapping. Inputs: config, a name
  # or mapping with class name and config; custom_objects, optional mapping of extra names to
  # user classes. It first refreshes the registry if the TF version changed, then delegates to
  # the generic Keras deserializer using the registry. Returns the initializer instance; the
  # generic deserializer raises a value error for unknown names.
  """Return an `Initializer` object from its config."""
  populate_deserializable_objects()
  return generic_utils.deserialize_keras_object(
      config,
      module_objects=LOCAL.ALL_OBJECTS,
      custom_objects=custom_objects,
      printable_module_name='initializer')


def get(identifier):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring examples call deserialize rather than get, and a string is
  # needlessly re-converted to a string before deserializing.
  # Resolves a flexible identifier to an initializer instance. Input: nothing (returns the null
  # value), a mapping (deserialized), a string name (deserialized), a class (instantiated with no
  # arguments) or any other callable (returned unchanged). Raises a value error for any other
  # type. No side effects beyond possible registry refresh.
  """Retrieve a Keras initializer by the identifier.

  The `identifier` may be the string name of a initializers function or class (
  case-sensitively).

  >>> identifier = 'Ones'
  >>> tf.keras.initializers.deserialize(identifier)
  <...keras.initializers.initializers_v2.Ones...>

  You can also specify `config` of the initializer to this function by passing
  dict containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Initializer` class.

  >>> cfg = {'class_name': 'Ones', 'config': {}}
  >>> tf.keras.initializers.deserialize(cfg)
  <...keras.initializers.initializers_v2.Ones...>

  In the case that the `identifier` is a class, this method will return a new
  instance of the class by its constructor.

  Args:
    identifier: String or dict that contains the initializer name or
      configurations.

  Returns:
    Initializer instance base on the input identifier.

  Raises:
    ValueError: If the input identifier is not a supported type or in a bad
      format.
  """

  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    if inspect.isclass(identifier):
      identifier = identifier()
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier: ' +
                     str(identifier))
