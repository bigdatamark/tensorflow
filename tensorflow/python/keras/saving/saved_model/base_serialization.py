# [MODULE DOC]
# HELPER: helper module for the Keras SavedModel serialization code (save and load paths).
# Defines the abstract base class SavedModelSaver, which declares what a Keras object must
# provide to be written into a SavedModel: an object identifier string, JSON metadata,
# extra trackable objects and extra functions. Concrete savers for layers, models, networks
# and metrics subclass it in sibling modules. No import-time side effects besides imports.
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
"""Helper classes that list&validate all attributes to serialize to SavedModel."""

import abc

from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils


class SavedModelSaver(object, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class (abstract metaclass) describing how one Keras object is serialized
  # into a SavedModel. Holds the wrapped object in the attribute obj. Subclasses supply the
  # object identifier, python properties for metadata, extra objects and extra functions.
  # Instances are created by the saving machinery, one per Keras object being exported.
  """Saver defining the methods and properties used to serialize Keras objects.
  """

  def __init__(self, obj):
    # [FUNCTION DOC]
    # Constructor. Takes obj, the Keras object (layer, model, metric, ...) to be serialized,
    # and stores it on the instance. Returns nothing; no side effects or exceptions.
    self.obj = obj

  @abc.abstractproperty
  def object_identifier(self):
    # [FUNCTION DOC]
    # Abstract property: subclasses return the string identifying the object type in the
    # SavedModel proto, used at load time to choose a reviver. The base version raises a
    # not-implemented error.
    # KNOWN ISSUE: uses the deprecated abstractproperty decorator instead of stacking
    # property over abstractmethod; behavior is equivalent.
    """String stored in object identifier field in the SavedModel proto.

    Returns:
      A string with the object identifier, which is used at load time.
    """
    raise NotImplementedError

  @property
  def tracking_metadata(self):
    # [FUNCTION DOC]
    # Property returning the metadata string stored in the SavedModel proto: the python
    # properties mapping encoded as JSON using the custom Keras JSON encoder. Takes no
    # inputs. Raises whatever the encoder or python_properties raises (for example a
    # not-implemented error in an incomplete subclass). No side effects.
    """String stored in metadata field in the SavedModel proto.

    Returns:
      A serialized JSON storing information necessary for recreating this layer.
    """
    # TODO(kathywu): check that serialized JSON can be loaded (e.g., if an
    # object is in the python property)
    return json_utils.Encoder().encode(self.python_properties)

  def trackable_children(self, serialization_cache):
    # [FUNCTION DOC]
    # Returns a mapping of attribute names to trackable children (objects and functions) to
    # save. Input serialization_cache is the per-export cache mapping shared by all objects.
    # Returns an empty mapping when saving traces is disabled; otherwise merges the results
    # of objects_to_serialize and functions_to_serialize (the objects mapping is updated in
    # place, so a subclass returning a shared mapping would see it mutated).
    """Lists all Trackable children connected to this object."""
    if not utils.should_save_traces():
      return {}

    children = self.objects_to_serialize(serialization_cache)
    children.update(self.functions_to_serialize(serialization_cache))
    return children

  @abc.abstractproperty
  def python_properties(self):
    # [FUNCTION DOC]
    # Abstract property: subclasses return a JSON-serializable mapping of python properties
    # that is saved as metadata and used at load time to rebuild the object. The base
    # version raises a not-implemented error.
    """Returns dictionary of python properties to save in the metadata.

    This dictionary must be serializable and deserializable to/from JSON.

    When loading, the items in this dict are used to initialize the object and
    define attributes in the revived object.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def objects_to_serialize(self, serialization_cache):
    # [FUNCTION DOC]
    # Abstract method: subclasses return a mapping of attribute names to extra trackable
    # objects to save. Input serialization_cache is the per-export cache mapping. The base
    # version raises a not-implemented error.
    """Returns dictionary of extra checkpointable objects to serialize.

    See `functions_to_serialize` for an explanation of this function's
    effects.

    Args:
      serialization_cache: Dictionary passed to all objects in the same object
        graph during serialization.

    Returns:
        A dictionary mapping attribute names to checkpointable objects.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def functions_to_serialize(self, serialization_cache):
    # [FUNCTION DOC]
    # Abstract method: subclasses return a mapping of attribute names to extra function
    # objects (tf functions or concrete functions) to save. Input serialization_cache is the
    # per-export cache mapping, which should be used to avoid regenerating functions. The
    # base version raises a not-implemented error.
    """Returns extra functions to include when serializing a Keras object.

    Normally, when calling exporting an object to SavedModel, only the
    functions and objects defined by the user are saved. For example:

    ```
    obj = tf.Module()
    obj.v = tf.Variable(1.)

    @tf.function
    def foo(...): ...

    obj.foo = foo

    w = tf.Variable(1.)

    tf.saved_model.save(obj, 'path/to/saved/model')
    loaded = tf.saved_model.load('path/to/saved/model')

    loaded.v  # Variable with the same value as obj.v
    loaded.foo  # Equivalent to obj.foo
    loaded.w  # AttributeError
    ```

    Assigning trackable objects to attributes creates a graph, which is used for
    both checkpointing and SavedModel serialization.

    When the graph generated from attribute tracking is insufficient, extra
    objects and functions may be added at serialization time. For example,
    most models do not have their call function wrapped with a @tf.function
    decorator. This results in `model.call` not being saved. Since Keras objects
    should be revivable from the SavedModel format, the call function is added
    as an extra function to serialize.

    This function and `objects_to_serialize` is called multiple times when
    exporting to SavedModel. Please use the cache to avoid generating new
    functions and objects. A fresh cache is created for each SavedModel export.

    Args:
      serialization_cache: Dictionary passed to all objects in the same object
        graph during serialization.

    Returns:
        A dictionary mapping attribute names to `Function` or
        `ConcreteFunction`.
    """
    raise NotImplementedError
