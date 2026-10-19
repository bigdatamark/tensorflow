# [MODULE DOC]
# HELPER: helper module for writing and reading the Keras layer metadata JSON in SavedModels.
# Provides a JSON encoder class that understands tensor shapes, tuples, type specs, numpy
# values, enums and similar objects, a decode function that reverses those encodings, and
# the get_json_type fallback converter. Tuples are tagged so they survive a round trip.
# No import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for creating and loading the Layer metadata for SavedModel.

These are required to retain the original format of the build input shape, since
layers and models may have different build behaviors depending on if the shape
is a list, tuple, or TensorShape. For example, Network.build() will create
separate inputs if the given input_shape is a list, and will create a single
input if the given shape is a tuple.
"""

import collections
import enum
import json
import numpy as np
import wrapt

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal


class Encoder(json.JSONEncoder):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says encoder and decoder, but this class only encodes;
  # decoding is done by the separate module-level decode function.
  # JSON encoder subclass used to write Keras metadata. Converts tuples into tagged mappings
  # (so they can be restored) and tensor shapes and other framework objects into plain JSON
  # structures. Used by creating an instance and calling its encode method.
  """JSON encoder and decoder that handles TensorShapes and tuples."""

  def default(self, obj):  # pylint: disable=method-hidden
    # [FUNCTION DOC]
    # Hook called by the JSON encoder for objects it cannot serialize. Input obj is the
    # object. A tensor shape becomes a mapping with class name TensorShape and its list of
    # dimensions (nothing/null if the rank is unknown); anything else is delegated to
    # get_json_type, which may raise a type error or value error. No side effects.
    """Encodes objects for types that aren't handled by the default encoder."""
    if isinstance(obj, tensor_shape.TensorShape):
      items = obj.as_list() if obj.rank is not None else None
      return {'class_name': 'TensorShape', 'items': items}
    return get_json_type(obj)

  def encode(self, obj):
    # [FUNCTION DOC]
    # Override of the standard encode. Input obj is any structure of nested lists, tuples and
    # mappings. First rewrites all tuples into tagged mappings (so decoding can restore
    # them), then encodes the result as a JSON string, which is returned.
    return super(Encoder, self).encode(_encode_tuple(obj))


def _encode_tuple(x):
  # [FUNCTION DOC]
  # HELPER: preprocessing step for Encoder.encode.
  # Recursively copies a nested structure x of tuples, lists and mappings, replacing each
  # tuple by a mapping tagged with the class name for tuples holding its encoded items.
  # Other values are returned unchanged. Does not mutate x.
  if isinstance(x, tuple):
    return {'class_name': '__tuple__',
            'items': tuple(_encode_tuple(i) for i in x)}
  elif isinstance(x, list):
    return [_encode_tuple(i) for i in x]
  elif isinstance(x, dict):
    return {key: _encode_tuple(value) for key, value in x.items()}
  else:
    return x


def decode(json_string):
  # [FUNCTION DOC]
  # Parses json_string (a JSON text) and returns the resulting structure, restoring tensor
  # shapes, type specs, tuples and the ellipsis value via the decoding helper applied to
  # every mapping. Raises a JSON decode error on malformed input. No side effects.
  return json.loads(json_string, object_hook=_decode_helper)


def _decode_helper(obj):
  # [FUNCTION DOC]
  # HELPER: object hook for decode.
  # Input obj is a parsed JSON value. If it is a mapping carrying a class name tag, converts
  # it back: TensorShape to a tensor shape, TypeSpec to an instance looked up in the type
  # spec registry and deserialized, tuple tag to a tuple, ellipsis tag to the ellipsis
  # value. Other values are returned unchanged. An unregistered type spec name may raise
  # from the registry lookup.
  """A decoding helper that is TF-object aware."""
  if isinstance(obj, dict) and 'class_name' in obj:
    if obj['class_name'] == 'TensorShape':
      return tensor_shape.TensorShape(obj['items'])
    elif obj['class_name'] == 'TypeSpec':
      return type_spec_registry.lookup(obj['type_spec'])._deserialize(  # pylint: disable=protected-access
          _decode_helper(obj['serialized']))
    elif obj['class_name'] == '__tuple__':
      return tuple(_decode_helper(i) for i in obj['items'])
    elif obj['class_name'] == '__ellipsis__':
      return Ellipsis
  return obj


def get_json_type(obj):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the branch for python types is effectively dead, because classes are
  # callable and are already handled by the earlier callable branch (same result: the name).
  # Converts an arbitrary object into a JSON-serializable structure, trying in order:
  # objects with a configuration method (class name plus config), numpy values (arrays to
  # lists, scalars to python values), callables and types (their name), dimensions (value),
  # tensor shapes (list), dtypes (name), mappings (plain dict), ellipsis (tagged mapping),
  # wrapt proxies (the wrapped object, returned unconverted), type specs (tagged mapping
  # with registered name and serialized form; raises a value error if unregistered), and
  # enum members (value). Otherwise raises a type error. Pure function, no side effects.
  """Serializes any object to a JSON-serializable structure.

  Args:
      obj: the object to serialize

  Returns:
      JSON-serializable structure representing `obj`.

  Raises:
      TypeError: if `obj` cannot be serialized.
  """
  # if obj is a serializable Keras class instance
  # e.g. optimizer, layer
  if hasattr(obj, 'get_config'):
    return {'class_name': obj.__class__.__name__, 'config': obj.get_config()}

  # if obj is any numpy type
  if type(obj).__module__ == np.__name__:
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return obj.item()

  # misc functions (e.g. loss function)
  if callable(obj):
    return obj.__name__

  # if obj is a python 'type'
  if type(obj).__name__ == type.__name__:
    return obj.__name__

  if isinstance(obj, tensor_shape.Dimension):
    return obj.value

  if isinstance(obj, tensor_shape.TensorShape):
    return obj.as_list()

  if isinstance(obj, dtypes.DType):
    return obj.name

  if isinstance(obj, collections.abc.Mapping):
    return dict(obj)

  if obj is Ellipsis:
    return {'class_name': '__ellipsis__'}

  if isinstance(obj, wrapt.ObjectProxy):
    return obj.__wrapped__

  if isinstance(obj, internal.TypeSpec):
    try:
      type_spec_name = type_spec_registry.get_name(type(obj))
      return {'class_name': 'TypeSpec', 'type_spec': type_spec_name,
              'serialized': obj._serialize()}  # pylint: disable=protected-access
    except ValueError:
      raise ValueError('Unable to serialize {} to JSON, because the TypeSpec '
                       'class {} has not been registered.'
                       .format(obj, type(obj)))

  if isinstance(obj, enum.Enum):
    return obj.value

  raise TypeError('Not JSON Serializable:', obj)
