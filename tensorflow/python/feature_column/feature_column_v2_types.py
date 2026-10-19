# [MODULE DOC]
# Defines the abstract base class FeatureColumn for the tf.feature_column API, kept in its own module
# so that other feature column modules and serialization code can share the type without cycles.
# Exposes one class, exported under the internal feature-column namespace for version 2 only.
# Import-time side effect: the class is registered with the API export decorator. No flags or
# global state are created.
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
"""Types specific to tf.feature_column."""
import abc

from tensorflow.python.util.tf_export import tf_export


@tf_export('__internal__.feature_column.FeatureColumn', v1=[])
class FeatureColumn(object, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class for all feature columns. A feature column describes how one named input
  # feature (or a combination of other columns) is parsed from examples and transformed into a
  # tensor. It holds no state itself; concrete subclasses supply the name, parsing spec, parent
  # dependencies, transformation logic and optional serialization config. Uses an abstract-base
  # metaclass, so subclasses must override the abstract members. Registered for API export.
  """Represents a feature column abstraction.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  To distinguish between the concept of a feature family and a specific binary
  feature within a family, we refer to a feature family like "country" as a
  feature column. For example, we can have a feature in a `tf.Example` format:
    {key: "country",  value: [ "US" ]}
  In this example the value of feature is "US" and "country" refers to the
  column of the feature.

  This class is an abstract class. Users should not create instances of this.
  """

  @abc.abstractproperty
  def name(self):
    # [FUNCTION DOC]
    # Abstract property giving the column's string name, used for naming and keys. Subclasses
    # must supply it; the base body does nothing and returns nothing.
    """Returns string. Used for naming."""
    pass

  def __lt__(self, other):
    # [FUNCTION DOC]
    # Less-than comparison so columns can be sorted. Input: any other object. Output: true when
    # the text form of this column sorts before the text form of the other object. No side effects.
    """Allows feature columns to be sorted in Python 3 as they are in Python 2.

    Feature columns need to occasionally be sortable, for example when used as
    keys in a features dictionary passed to a layer.

    In CPython, `__lt__` must be defined for all objects in the
    sequence being sorted.

    If any objects in the sequence being sorted do not have an `__lt__` method
    compatible with feature column objects (such as strings), then CPython will
    fall back to using the `__gt__` method below.
    https://docs.python.org/3/library/stdtypes.html#list.sort

    Args:
      other: The other object to compare to.

    Returns:
      True if the string representation of this object is lexicographically less
      than the string representation of `other`. For FeatureColumn objects,
      this looks like "<__main__.FeatureColumn object at 0xa>".
    """
    return str(self) < str(other)

  def __gt__(self, other):
    # [FUNCTION DOC]
    # Greater-than comparison used by the sort machinery when the other operand lacks a
    # compatible less-than. Input: any other object. Output: true when the text form of this
    # column sorts after the text form of the other object. No side effects.
    """Allows feature columns to be sorted in Python 3 as they are in Python 2.

    Feature columns need to occasionally be sortable, for example when used as
    keys in a features dictionary passed to a layer.

    `__gt__` is called when the "other" object being compared during the sort
    does not have `__lt__` defined.
    Example:
    ```
    # __lt__ only class
    class A():
      def __lt__(self, other): return str(self) < str(other)

    a = A()
    a < "b" # True
    "0" < a # Error

    # __lt__ and __gt__ class
    class B():
      def __lt__(self, other): return str(self) < str(other)
      def __gt__(self, other): return str(self) > str(other)

    b = B()
    b < "c" # True
    "0" < b # True
    ```

    Args:
      other: The other object to compare to.

    Returns:
      True if the string representation of this object is lexicographically
      greater than the string representation of `other`. For FeatureColumn
      objects, this looks like "<__main__.FeatureColumn object at 0xa>".
    """
    return str(self) > str(other)

  @abc.abstractmethod
  def transform_feature(self, transformation_cache, state_manager):
    # [FUNCTION DOC]
    # Abstract method that builds the column's intermediate representation (usually a tensor)
    # from raw features or parent columns. Inputs: a transformation cache used to fetch raw or
    # parent-column tensors, and a state manager used to create or access resources such as lookup
    # tables. Output: the transformed tensor. The base body does nothing and returns nothing.
    """Returns intermediate representation (usually a `Tensor`).

    Uses `transformation_cache` to create an intermediate representation
    (usually a `Tensor`) that other feature columns can use.

    Example usage of `transformation_cache`:
    Let's say a Feature column depends on raw feature ('raw') and another
    `FeatureColumn` (input_fc). To access corresponding `Tensor`s,
    transformation_cache will be used as follows:

    ```python
    raw_tensor = transformation_cache.get('raw', state_manager)
    fc_tensor = transformation_cache.get(input_fc, state_manager)
    ```

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Transformed feature `Tensor`.
    """
    pass

  @abc.abstractproperty
  def parse_example_spec(self):
    # [FUNCTION DOC]
    # Abstract property returning the example-parsing specification as a mapping from feature
    # keys to fixed-length, variable-length or other parsing feature descriptions, suitable for
    # the example parsing op. Subclasses build it from their own key plus their parents' specs.
    # The base body does nothing and returns nothing.
    """Returns a `tf.Example` parsing spec as dict.

    It is used for get_parsing_spec for `tf.io.parse_example`. Returned spec is
    a dict from keys ('string') to `VarLenFeature`, `FixedLenFeature`, and other
    supported objects. Please check documentation of `tf.io.parse_example` for
    all supported spec objects.

    Let's say a Feature column depends on raw feature ('raw') and another
    `FeatureColumn` (input_fc). One possible implementation of
    parse_example_spec is as follows:

    ```python
    spec = {'raw': tf.io.FixedLenFeature(...)}
    spec.update(input_fc.parse_example_spec)
    return spec
    ```
    """
    pass

  def create_state(self, state_manager):
    # [FUNCTION DOC]
    # Hook for creating state (variables, lookup tables) the column needs. Input: a state manager
    # used to create or access such resources. Default implementation does nothing and returns
    # nothing; stateful subclasses override it.
    """Uses the `state_manager` to create state for the FeatureColumn.

    Args:
      state_manager: A `StateManager` to create / access resources such as
        lookup tables and variables.
    """
    pass

  @abc.abstractproperty
  def _is_v2_column(self):
    # [FUNCTION DOC]
    # HELPER: internal compatibility flag used when mixing old and new column implementations.
    # Abstract property reporting whether the column conforms fully to the new column API.
    # Subclasses supply a boolean. The base body does nothing and returns nothing.
    """Returns whether this FeatureColumn is fully conformant to the new API.

    This is needed for composition type cases where an EmbeddingColumn etc.
    might take in old categorical columns as input and then we want to use the
    old API.
    """
    pass

  @abc.abstractproperty
  def parents(self):
    # [FUNCTION DOC]
    # Abstract property returning the list of immediate dependencies: raw feature key strings
    # and/or other feature column objects. Subclasses supply it. The base body does nothing and
    # returns nothing.
    """Returns a list of immediate raw feature and FeatureColumn dependencies.

    For example:
    # For the following feature columns
    a = numeric_column('f1')
    c = crossed_column(a, 'f2')
    # The expected parents are:
    a.parents = ['f1']
    c.parents = [a, 'f2']
    """
    pass

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a serializable mapping describing this column so it can be rebuilt later with the
    # matching from-config method. Delegates to the private config hook, so it raises a
    # not-implemented error unless the subclass overrides that hook. No side effects.
    """Returns the config of the feature column.

    A FeatureColumn config is a Python dictionary (serializable) containing the
    configuration of a FeatureColumn. The same FeatureColumn can be
    reinstantiated later from this configuration.

    The config of a feature column does not include information about feature
    columns depending on it nor the FeatureColumn class name.

    Example with (de)serialization practices followed in this file:
    ```python
    class SerializationExampleFeatureColumn(
        FeatureColumn, collections.namedtuple(
            'SerializationExampleFeatureColumn',
            ('dimension', 'parent', 'dtype', 'normalizer_fn'))):

      def get_config(self):
        # Create a dict from the namedtuple.
        # Python attribute literals can be directly copied from / to the config.
        # For example 'dimension', assuming it is an integer literal.
        config = dict(zip(self._fields, self))

        # (De)serialization of parent FeatureColumns should use the provided
        # (de)serialize_feature_column() methods that take care of de-duping.
        config['parent'] = serialize_feature_column(self.parent)

        # Many objects provide custom (de)serialization e.g: for tf.DType
        # tf.DType.name, tf.as_dtype() can be used.
        config['dtype'] = self.dtype.name

        # Non-trivial dependencies should be Keras-(de)serializable.
        config['normalizer_fn'] = generic_utils.serialize_keras_object(
            self.normalizer_fn)

        return config

      @classmethod
      def from_config(cls, config, custom_objects=None, columns_by_name=None):
        # This should do the inverse transform from `get_config` and construct
        # the namedtuple.
        kwargs = config.copy()
        kwargs['parent'] = deserialize_feature_column(
            config['parent'], custom_objects, columns_by_name)
        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
        kwargs['normalizer_fn'] = generic_utils.deserialize_keras_object(
          config['normalizer_fn'], custom_objects=custom_objects)
        return cls(**kwargs)

    ```
    Returns:
      A serializable Dict that can be used to deserialize the object with
      from_config.
    """
    return self._get_config()

  def _get_config(self):
    # [FUNCTION DOC]
    # HELPER: private hook behind the public config getter.
    # Default implementation always raises a not-implemented error; subclasses override it to
    # return their serializable configuration mapping.
    raise NotImplementedError('Must be implemented in subclasses.')

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # Class method that rebuilds a column from a config mapping produced by the config getter.
    # Inputs: the config mapping; optional mapping of names to custom classes or functions used
    # during deserialization; optional mapping of column names to already built columns, used to
    # avoid duplicates. Output: the new column. Delegates to the private from-config hook, so it
    # raises a not-implemented error unless the subclass overrides that hook.
    """Creates a FeatureColumn from its config.

    This method should be the reverse of `get_config`, capable of instantiating
    the same FeatureColumn from the config dictionary. See `get_config` for an
    example of common (de)serialization practices followed in this file.

    TODO(b/118939620): This is a private method until consensus is reached on
    supporting object deserialization deduping within Keras.

    Args:
      config: A Dict config acquired with `get_config`.
      custom_objects: Optional dictionary mapping names (strings) to custom
        classes or functions to be considered during deserialization.
      columns_by_name: A Dict[String, FeatureColumn] of existing columns in
        order to avoid duplication. Should be passed to any calls to
        deserialize_feature_column().

    Returns:
      A FeatureColumn for the input config.
    """
    return cls._from_config(config, custom_objects, columns_by_name)

  @classmethod
  def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # HELPER: private class-method hook behind the public from-config method.
    # Takes the same arguments (config mapping, optional custom objects, optional existing columns
    # by name) and always raises a not-implemented error; subclasses override it to rebuild
    # themselves from a config.
    raise NotImplementedError('Must be implemented in subclasses.')
