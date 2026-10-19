# [MODULE DOC]
# Implements user-defined extension types: the public base class for classes whose instances are
# immutable collections of typed fields that work with graphs, functions, control flow, batching and
# saved models. Provides the metaclass that, at class creation, validates field annotations, builds
# a constructor and generates a nested companion type specification class; the base specification
# class; batchable variants and the default batch encoder; pack and unpack helpers converting between
# field-wise storage and a single opaque variant; a dictionary-conversion helper; anonymous
# extension types used as a fallback when a saved model refers to an unregistered type; a
# reinterpret helper to convert between compatible types; and a codec registered with the structured
# value coder for serializing specifications.
# Import-time side effects: registers the specification codec with the nested structure coder and
# registers the anonymous specification type under a fixed name; public classes and functions are
# exported to the experimental API namespace.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""User-defined ExtensionType classes."""

import abc
import typing
import warnings

import typing_extensions

from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

# Attribute used to keep track of when we're inside a user-defined constructor
# (in which case the fields of `self` may be modified).
_IN_CONSTRUCTOR = '_tf_extension_type_in_constructor'

_MUTABLE_KERAS_PROPERTIES = [
    # Keras uses _keras_mask property to pass the mask around
    '_keras_mask',
]


# ==============================================================================
# Utility functions
# ==============================================================================
def _create_object_from_type_and_dict(cls, obj_dict):
  # [FUNCTION DOC]
  # HELPER: builds extension type and specification objects without running their constructors.
  # Inputs: a class and a mapping of attribute names to values. Creates a bare instance of the class
  # without calling its constructor, copies all mapping entries into its attribute dictionary, and
  # returns it. (A sequence of name and value pairs also works.) No validation is performed.
  """Creates an object, bypassing the constructor.

  Creates an object of type `cls`, whose `__dict__` is updated to contain
  `obj_dict`.

  Args:
    cls: The type of the new object.
    obj_dict: A `Mapping` that should be used to initialize the new object's
      `__dict__`.

  Returns:
    An object of type `cls`.
  """
  value = object.__new__(cls)
  value.__dict__.update(obj_dict)
  return value


# ==============================================================================
# Metaclass for tf.ExtensionType
# ==============================================================================
class ExtensionTypeMetaclass(abc.ABCMeta):
  # [CLASS DOC]
  # Metaclass of the extension type base class. When a user subclass is created it validates the field
  # annotations, installs a constructor (wrapping a user-defined one or generating one from the fields)
  # and creates the nested specification class. Framework classes opt out by defining a marker class
  # attribute in their own namespace (the marker is not inherited).
  """Metaclass for tf.ExtensionType types."""

  def __init__(cls, name, bases, namespace):
    # [FUNCTION DOC]
    # Runs after a class statement creates the class. Inputs: the new class, its name, its base classes and
    # its namespace mapping. Unless the namespace itself carries the do-not-transform marker set to true,
    # validates the field annotations (may raise a value error), adds a constructor, and builds the nested
    # specification class; then calls the base metaclass initializer. Mutates the new class.
    # Don't transform base classes that are part of the framework -- only
    # transform user classes.  We identify classes that are part of the
    # framework by setting '_tf_extension_type_do_not_transform_this_class=True'
    # in the class definition.  (Note: we check for this in the class namespace,
    # so it is *not* ineherited.)
    if not namespace.get(
        '_tf_extension_type_do_not_transform_this_class', False
    ):
      _check_field_annotations(cls)
      _add_extension_type_constructor(cls)
      _add_type_spec(cls)
    super(ExtensionTypeMetaclass, cls).__init__(name, bases, namespace)


# ==============================================================================
# Base class for user-defined types
# ==============================================================================
@tf_export('experimental.ExtensionType')
class ExtensionType(
    composite_tensor.CompositeTensor, metaclass=ExtensionTypeMetaclass
):
  # [CLASS DOC]
  # Public base class for user-defined extension types, a kind of composite tensor. Subclasses declare
  # typed fields as class annotations (optionally with defaults); instances are immutable outside of a
  # custom constructor, support equality and representation, and expose a type specification describing
  # their static part. Instances may be in a "packed" form where all fields live in one variant tensor,
  # in which case attribute lookups unpack on demand. The base class itself is abstract and cannot be
  # instantiated; it carries a marker so the metaclass does not transform it. Class-level state: a
  # per-class cache of the resolved field list and a cache for the instance specification.
  """Base class for TensorFlow `ExtensionType` classes.

  Tensorflow `ExtensionType` classes are specialized Python classes that can be
  used transparently with TensorFlow -- e.g., they can be used with ops
  such as `tf.cond` or `tf.while_loop` and used as inputs or outputs for
  `tf.function` and Keras layers.

  New `ExtensionType` classes are defined by creating a subclass of
  `tf.ExtensionType` that
  contains type annotations for all instance variables.  The following type
  annotations are supported:

  Type                      | Example
  ------------------------- | --------------------------------------------
  Python integers           | `i: int`
  Python floats             | `f: float`
  Python strings            | `s: str`
  Python booleans           | `b: bool`
  Python None               | `n: None`
  Python tuple              | `params: tuple[int, float, int, int]`
  Python tuple w/ Ellipsis  | `lengths: tuple[int, ...]`
  Tensors                   | `t: tf.Tensor`
  Composite Tensors         | `rt: tf.RaggedTensor`
  Extension Types           | `m: MyMaskedTensor`
  Tensor shapes             | `shape: tf.TensorShape`
  Tensor dtypes             | `dtype: tf.DType`
  Type unions               | `length: typing.Union[int, float]`
  Tuples                    | `params: typing.Tuple[int, float, int, int]`
  Tuples w/ Ellipsis        | `lengths: typing.Tuple[int, ...]`
  Mappings                  | `tags: typing.Mapping[str, str]`

  Fields annotated with `typing.Mapping` will be stored using an immutable
  mapping type.

  ExtensionType values are immutable -- i.e., once constructed, you can not
  modify or delete any of their instance members.

  ### Examples

  >>> class MaskedTensor(ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor

  >>> class Toy(ExtensionType):
  ...   name: str
  ...   price: tensor.Tensor
  ...   features: typing.Mapping[str, tf.Tensor]

  >>> class ToyStore(ExtensionType):
  ...   name: str
  ...   toys: typing.Tuple[Toy, ...]
  """

  # Let the metaclass know that it should *not* transform this class (since
  # this class is part of the ExtensionType framework, and not a user class).
  _tf_extension_type_do_not_transform_this_class = True

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor of the abstract base class. Accepts and ignores any arguments. Raises an assertion
    # error if the object being created is exactly the base class rather than a subclass. Subclass
    # constructors generated by the metaclass call it indirectly (or not at all); it sets no state.
    if type(self) is ExtensionType:  # pylint: disable=unidiomatic-typecheck
      raise AssertionError(
          'Cannot create an instance of ExtensionType '
          'because ExtensionType is an abstract base class.'
      )

  # This class variable is used to cache the return value for
  # _tf_extension_type_fields.
  _tf_extension_type_cached_fields = None

  @classmethod
  def _tf_extension_type_fields(cls):  # pylint: disable=no-self-argument
    # [FUNCTION DOC]
    # Class method returning the ordered fields of this type. Collects type hints of the class (leaving
    # out annotation metadata wrappers) and builds one field description per annotated name, carrying the
    # name, its declared type and the class-level default (or a no-default marker). If some forward
    # reference cannot be resolved yet, falls back to merging raw annotations from the class hierarchy
    # from base to derived and does not cache the result; otherwise caches the tuple on the class (not
    # inherited from base classes). Returns a tuple of field descriptions. Side effect: sets the cache.
    """An ordered list describing the fields of this ExtensionType.

    Returns:
      A list of `ExtensionTypeField` objects.  Forward references are resolved
      if possible, or left unresolved otherwise.
    """
    if '_tf_extension_type_cached_fields' in cls.__dict__:  # do not inherit.
      return cls._tf_extension_type_cached_fields

    try:
      # Using include_extras=False will replace all Annotated[T, ...] with T.
      # The typing_extensions module is used since this is only supported in
      # Python 3.9.
      type_hints = typing_extensions.get_type_hints(cls, include_extras=False)
      ok_to_cache = True  # all forward references have been resolved.
    except (NameError, AttributeError):
      # Unresolved forward reference -- gather type hints manually.
      # * NameError comes from an annotation like `Foo` where class
      #   `Foo` hasn't been defined yet.
      # * AttributeError comes from an annotation like `foo.Bar`, where
      #   the module `foo` exists but `Bar` hasn't been defined yet.
      # Note: If a user attempts to instantiate a `ExtensionType` type that
      # still has unresolved forward references (e.g., because of a typo or a
      # missing import), then the constructor will raise an exception.
      type_hints = {}
      for base in reversed(cls.__mro__):
        type_hints.update(base.__dict__.get('__annotations__', {}))
      ok_to_cache = False

    fields = []
    for name, value_type in type_hints.items():
      default = getattr(
          cls, name, extension_type_field.ExtensionTypeField.NO_DEFAULT
      )
      fields.append(
          extension_type_field.ExtensionTypeField(name, value_type, default)
      )
    fields = tuple(fields)

    if ok_to_cache:
      cls._tf_extension_type_cached_fields = fields

    return fields

  @classmethod
  def _tf_extension_type_has_field(cls, name):
    # [FUNCTION DOC]
    # Class method telling whether a name is one of the declared fields. Input: attribute name.
    # Returns true if any field description has that name, otherwise false.
    return any(name == field.name for field in cls._tf_extension_type_fields())

  def _tf_extension_type_convert_fields(self):
    # [FUNCTION DOC]
    # Converts the stored field values of this instance, in place, to the canonical forms required by
    # their declared types (for example lists to tuples, plain mappings to immutable mappings, values
    # converted to tensors where annotated) by delegating to the field helper module, which also raises
    # a type or value error when a value does not fit its annotation. Returns nothing.
    extension_type_field.convert_fields(
        self._tf_extension_type_fields(), self.__dict__
    )

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a readable string: the qualified class name followed by all fields as name equals
    # representation pairs separated by commas, in declaration order.
    fields = ', '.join(
        [
            f'{field.name}={getattr(self, field.name)!r}'
            for field in self._tf_extension_type_fields()
        ]
    )
    return f'{type(self).__qualname__}({fields})'

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Enforces immutability. Inputs: attribute name and value. Allowed only for the special mask
    # property used by the neural network library, or while the instance is inside a user-defined
    # constructor and the name is a declared field; then stores the value directly in the attribute
    # dictionary. Otherwise raises an attribute error saying mutation is allowed only in the constructor.
    if name in _MUTABLE_KERAS_PROPERTIES or (
        hasattr(self, _IN_CONSTRUCTOR)
        and self._tf_extension_type_has_field(name)
    ):
      self.__dict__[name] = value
    else:
      raise AttributeError(
          f'Cannot mutate attribute `{name}` '
          'outside the custom constructor of ExtensionType.'
      )

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Enforces immutability for deletion. Input: attribute name. Removes the attribute from the instance
    # dictionary only for the mask property or for a declared field inside a user constructor; otherwise
    # raises an attribute error saying mutation is allowed only in the custom constructor.
    if name in _MUTABLE_KERAS_PROPERTIES or (
        hasattr(self, _IN_CONSTRUCTOR)
        and self._tf_extension_type_has_field(name)
    ):
      del self.__dict__[name]
    else:
      raise AttributeError(
          f'Cannot mutate attribute `{name}` '
          'outside the custom constructor of ExtensionType.'
      )

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Fallback attribute lookup, called only when normal lookup fails. Input: attribute name. For the
    # mask property uses the default object lookup. If this instance is packed, unpacks it and looks the
    # name up on the unpacked copy (deliberately not cached, since caching could make a tensor created in
    # one conditional branch be used in another). Otherwise raises an attribute error naming the type.
    if name in _MUTABLE_KERAS_PROPERTIES:
      return object.__getattribute__(self, name)
    if '_tf_extension_type_packed_variant' in self.__dict__:
      # Note: it's *not* ok to cache the results of unpack() here.  In
      # particular, it would be nice if we could do something like
      # `self.__dict__.update(unpack(self).__dict__)`, but that (potentially)
      # violates an invariant required by the `cond` operation.  E.g., if we had
      # `tf.cond(lambda: x.foo, lambda: x.bar)`, then tensor `x.bar` used in the
      # "else" branch would be created by an op in the "then" branch (when
      # looking up `x.foo`); and that's not allowed.
      return getattr(unpack(self), name)

    raise AttributeError(
        f'{type(self).__name__!r} object has no attribute {name!r}'
    )

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Value equality that works with tensors. Input: another object. Returns plain false if the types
    # differ, the type specifications differ, or the flattened component lists have different lengths.
    # Otherwise returns a scalar boolean tensor that is true when, for every pair of components, shapes
    # are equal and all elements are equal (broadcast-equal values of different shapes are not equal).
    # Because of that the result may be a tensor rather than a Python boolean. Adds ops to the graph.
    if type(self) is not type(other):
      return False

    if self._type_spec != other._type_spec:
      return False

    self_tensors = nest.flatten(self, expand_composites=True)
    other_tensors = nest.flatten(other, expand_composites=True)
    if len(self_tensors) != len(other_tensors):
      return False
    conditions = []
    for t1, t2 in zip(self_tensors, other_tensors):
      conditions.append(
          math_ops.reduce_all(
              gen_math_ops.equal(
                  array_ops.shape(t1),
                  array_ops.shape(t2),
                  incompatible_shape_error=False,
              )
          )
      )
      # Explicitly check shape (values that have different shapes but broadcast
      # to the same value are considered non-equal).
      conditions.append(
          math_ops.reduce_all(
              gen_math_ops.equal(t1, t2, incompatible_shape_error=False)
          )
      )
    return math_ops.reduce_all(array_ops_stack.stack(conditions))

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Negation of equality. Input: another object. If the equality result is a tensor, returns its
    # logical negation as a tensor; otherwise returns the Python negation.
    eq = self.__eq__(other)
    if isinstance(eq, tensor.Tensor):
      return math_ops.logical_not(eq)
    else:
      return not eq

  def __validate__(self):
    # [FUNCTION DOC]
    # Hook for subclasses to override with extra validation after construction. The default does nothing.
    """Perform post-construction validation."""

  # This instance variable is used to cache the value for the _type_spec
  # property.
  _tf_extension_type_cached_type_spec = None

  @property
  def _type_spec(self):  # CompositeTensor API.
    # [FUNCTION DOC]
    # Read-only property implementing the composite tensor protocol: returns the type specification of
    # this value, built on first use from the value and cached in the instance dictionary. Asserts the
    # value is not packed (packed values always carry a cached specification). Side effect: caching.
    # Note: the TypeSpec contains all static (non-tensor) data from `self`.
    if self._tf_extension_type_cached_type_spec is None:
      assert not is_packed(self)  # Packed version always caches TypeSpec.
      self.__dict__['_tf_extension_type_cached_type_spec'] = (
          self.Spec.from_value(self)
      )
    return self._tf_extension_type_cached_type_spec


@tf_export('experimental.extension_type.as_dict')
def as_dict(value):
  # [FUNCTION DOC]
  # Converts an extension type instance to a mapping of field names to values. Input: an extension
  # type instance. Returns a new mapping in declaration order. Not recursive: nested extension types
  # stay as they are. Exported in the experimental extension type API.
  """Extracts the attributes of `value` and their values to a dict format.

  Unlike `dataclasses.asdict()`, this function is not recursive and in case of
  nested `ExtensionType` objects, only the top level object is converted to a
  dict.

  Args:
    value: An `ExtensionType` object.

  Returns:
    A dict that contains the attributes of `value` and their values.
  """
  return {
      field.name: getattr(value, field.name)
      for field in value._tf_extension_type_fields()  # pylint: disable=protected-access
  }


def pack(value):
  # [FUNCTION DOC]
  # Returns a copy of an extension type value whose fields are stored in one variant tensor. Input: an
  # extension type instance. If already packed, returns it unchanged. Otherwise encodes the value as a
  # variant, and builds a new object of the same type holding only that variant and the packed form
  # of the specification. Raises a value error if the type lacks a name so it cannot be encoded.
  """Returns a copy of `value` with fields packed in a single Variant.

  Args:
    value: An `ExtensionType` object.

  Returns:
    An `ExtensionType` object.
  """
  if is_packed(value):
    return value

  spec = value._type_spec._tf_extension_type_with_packed(True)  # pylint: disable=protected-access
  try:
    variant = composite_tensor_ops.composite_tensor_to_variants(value)
  except nested_structure_coder.NotEncodableError as e:
    # Note: the only time `_TypeSpecCodec.can_encode` returns False is if the
    # named type is not registered.  The default error message would simply
    # tell the user that there is no encoder for the object, so we provide
    # a more useful message letting them know how to register the type.
    raise ValueError(
        'ExtensionTypes must have a __name__ field in order to be packed.'
    ) from e

  return _create_object_from_type_and_dict(
      type(value),
      {
          '_tf_extension_type_cached_type_spec': spec,
          '_tf_extension_type_packed_variant': variant,
      },
  )


def unpack(value):
  # [FUNCTION DOC]
  # Inverse of packing. Input: an extension type instance. If it is not packed, returns it as is;
  # otherwise decodes the stored variant using the unpacked form of the cached specification and
  # returns the resulting value with individual fields.
  """Returns a copy of `value` with individual fields stored in __dict__.

  Args:
    value: An `ExtensionType` object.

  Returns:
    An `ExtensionType` object.
  """
  if not is_packed(value):
    return value

  # pylint: disable=protected-access
  variant = value._tf_extension_type_packed_variant
  spec = value._tf_extension_type_cached_type_spec
  spec = spec._tf_extension_type_with_packed(False)
  return composite_tensor_ops.composite_tensor_from_variant(variant, spec)


def is_packed(value):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the error message is missing a space after the comma, running its two sentences together.
  # Tells whether a value stores its fields in a single variant. Input: an extension type instance.
  # Returns true if the packed variant marker is in its attribute dictionary. Raises a value error if
  # the argument is not an extension type instance.
  """Returns true if `value`'s fields are packed in a single Variant."""
  if not isinstance(value, ExtensionType):
    raise ValueError(
        'Expected `value` to be an object of type ExtensionType,'
        f'got an instance of {type(value)}.'
    )
  return '_tf_extension_type_packed_variant' in value.__dict__


# ==============================================================================
# Base class for the tf.ExtensionType TypeSpecs
# ==============================================================================


@tf_export('experimental.ExtensionTypeSpec')
class ExtensionTypeSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # Base class of the type specifications generated for extension types. Its instance dictionary
  # holds the specification versions of the fields (tensors become tensor specifications, other
  # composite values their specifications, plain values stay as is), and a flag marks the packed
  # form. Implements the type specification protocol (serialize, components, rebuild from components),
  # immutability outside constructors, and pickling through the owning value type. The owning
  # extension type class is exposed as the value type attribute of generated subclasses.
  """Base class for tf.ExtensionType TypeSpec."""

  def _serialize(self):  # TypeSpec API.
    # [FUNCTION DOC]
    # Returns the serialized state: a tuple of name and value pairs in field order (the packed flag is
    # appended when set), where nested immutable mappings are converted to plain dictionaries.
    # Use a tuple of (name, value) pairs, to ensure we preserve field ordering.
    fields = [f.name for f in self._tf_extension_type_fields()]
    if self._tf_extension_type_is_packed:
      fields.append('_tf_extension_type_is_packed')
    return tuple(
        (f, _change_nested_mappings_to(self.__dict__[f], dict)) for f in fields
    )

  @classmethod
  def _deserialize(cls, state):  # TypeSpec API.
    # [FUNCTION DOC]
    # Class method inverse of serialization. Input: the serialized state. Converts nested plain
    # mappings to immutable ones and builds a specification instance of this class from the state
    # without running the constructor. Returns the instance.
    state = _change_nested_mappings_to(state, immutable_dict.ImmutableDict)
    return _create_object_from_type_and_dict(cls, state)

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickle support. Returns the module-level deserialization function plus the value type and the
    # serialized state, using the value type rather than the nested specification class because nested
    # classes cannot be pickled by default in older Python versions.
    # Use value_type instead of spec_type, as spec_type is a nested class.
    # Pickle support of nested class requires Pickle protocol version 4, which
    # is not enabled by default until py 3.8.
    #
    # https://www.python.org/dev/peps/pep-3154/#serializing-more-lookupable-objects
    # https://docs.python.org/3/library/pickle.html#pickle.DEFAULT_PROTOCOL
    return _deserialize_for_reduce, (self.value_type, self._serialize())

  def _to_components(self, value):  # TypeSpec API.
    # [FUNCTION DOC]
    # Splits a value into its tensor components. Input: a value of this type. For a packed specification
    # returns the single variant tensor. Otherwise returns a tuple of all tensors and composite tensors
    # found by flattening the value's fields in specification order (preserving field order).
    if self._tf_extension_type_is_packed:
      return value._tf_extension_type_packed_variant  # pylint: disable=protected-access

    tensor_or_composite = (tensor.Tensor, composite_tensor.CompositeTensor)
    # Retrieve fields by the order of spec dict to preserve field ordering. This
    # is needed as nest.flatten would sort dictionary entries by key.
    value_tuple = tuple(value.__dict__[key] for key in self.__dict__)
    return tuple(
        x
        for x in nest.flatten(value_tuple)
        if isinstance(x, tensor_or_composite)
    )

  def _from_components(self, components):  # TypeSpec API.
    # [FUNCTION DOC]
    # Rebuilds a value from components. Input: the component values. For a packed specification wraps
    # the given variant in a packed value. Otherwise substitutes the components, in order, for the
    # nested specifications among the fields, repacks the structure, and creates the value without calling
    # its constructor. Raises a value error if more components are given than the specification expects.
    if self._tf_extension_type_is_packed:
      return _create_object_from_type_and_dict(
          self.value_type,
          {
              '_tf_extension_type_cached_type_spec': self,
              '_tf_extension_type_packed_variant': components,
          },
      )

    spec_tuple = tuple(self.__dict__.values())
    components_iter = iter(components)
    flat = [
        next(components_iter) if isinstance(x, type_spec.TypeSpec) else x
        for x in nest.flatten(spec_tuple)
    ]
    if list(components_iter):
      raise ValueError(
          'Cannot build an ExtensionType instance from components '
          'because more components are provided than the number expected '
          'by the type spec.'
      )
    value_tuple = nest.pack_sequence_as(spec_tuple, flat)
    fields = dict(zip(self.__dict__.keys(), value_tuple))

    # Build the new value.  Bypass the constructor (__init__), in case the user
    # who defined the ExtensionType used a custom constructor.
    return _create_object_from_type_and_dict(self.value_type, fields)

  @property
  def _component_specs(self):  # TypeSpec API.
    # [FUNCTION DOC]
    # Read-only property: the specifications of the components. For a packed specification, a scalar
    # variant tensor specification; otherwise a tuple of every nested type specification found in the
    # fields, in order.
    if self._tf_extension_type_is_packed:
      return tensor.TensorSpec((), dtypes.variant)

    components = []

    def push_if_type_spec(x):
      # [FUNCTION DOC]
      # HELPER: nested callback for walking the fields. Appends its argument to the enclosing list of
      # components if the argument is a type specification; otherwise ignores it. Returns nothing.
      if isinstance(x, type_spec.TypeSpec):
        components.append(x)

    nest.map_structure(push_if_type_spec, tuple(self.__dict__.values()))
    return tuple(components)

  @classmethod
  def from_value(cls, value):
    # [FUNCTION DOC]
    # Class method building a specification from a value. Input: an extension type value. Returns the
    # value's cached specification if it has one. Otherwise replaces tensors in its fields with tensor
    # specifications (without names) and other composites with their specifications, drops the cached
    # field list entry, and builds a specification object of this class without calling its constructor.
    cached_spec = getattr(value, '_tf_extension_type_cached_type_spec', None)
    if cached_spec is not None:
      return cached_spec

    value_fields = value.__dict__
    spec_fields = nest.map_structure(_replace_tensor_with_spec, value_fields)
    spec_fields.pop('_tf_extension_type_cached_fields', None)
    return _create_object_from_type_and_dict(cls, spec_fields)

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Immutability guard. Inputs: name and value. Stores into the instance dictionary when inside a
    # user-defined constructor and the name is a field; delegates to the parent for the cached fixed
    # properties of type specifications; otherwise raises an attribute error.
    if (hasattr(self, _IN_CONSTRUCTOR)
        and self._tf_extension_type_has_field(name)):
      self.__dict__[name] = value
    elif name in type_spec.CACHED_FIXED_PROPERTIES:
      super().__setattr__(name, value)
    else:
      raise AttributeError(
          f'Cannot mutate attribute `{name}` '
          'outside the custom constructor of ExtensionTypeSpec.'
      )

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Deletes a field from the instance dictionary only inside a user-defined constructor and only for
    # declared fields; in every other case raises an attribute error saying mutation is not allowed.
    if hasattr(self, _IN_CONSTRUCTOR) and self._tf_extension_type_has_field(
        name
    ):
      del self.__dict__[name]
    else:
      raise AttributeError(
          f'Cannot mutate attribute `{name}` '
          'outside the custom constructor of ExtensionTypeSpec.'
      )

  def __validate__(self):
    # [FUNCTION DOC]
    # Hook for subclasses to add post-construction validation of the specification. Does nothing by default.
    """Perform post-construction validation."""

  @classmethod
  def _tf_extension_type_fields(cls):
    # [FUNCTION DOC]
    # Class method returning the field descriptions of the owning value type (delegates to it).
    return cls.value_type._tf_extension_type_fields()  # pylint: disable=protected-access

  @classmethod
  def _tf_extension_type_has_field(cls, name):
    # [FUNCTION DOC]
    # Class method telling whether the given name is one of the owning value type's fields.
    # Returns true or false.
    return any(name == field.name for field in cls._tf_extension_type_fields())

  def _tf_extension_type_convert_fields(self):
    # [FUNCTION DOC]
    # Converts the stored field values of this specification in place to the forms required for
    # specifications, using the field helper module; may raise type or value errors. Returns nothing.
    extension_type_field.convert_fields_for_spec(
        self._tf_extension_type_fields(), self.__dict__
    )

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a readable string: the qualified class name with the serialized fields as name and value
    # pairs.
    fields = ', '.join([f'{k}={v!r}' for (k, v) in self._serialize()])
    return f'{type(self).__qualname__}({fields})'

  _tf_extension_type_is_packed = False

  def _tf_extension_type_with_packed(self, value):
    # [FUNCTION DOC]
    # Returns a copy of this specification with the packed flag set to the given boolean. Input: boolean.
    # The original is not modified.
    """Returns a copy of this `TypeSpec` with `packed=value`.

    Args:
      value: A boolean value.

    Returns:
      A copy of `self` with `_tf_extension_type_is_packed=value`.
    """
    copy = _create_object_from_type_and_dict(type(self), self.__dict__)
    copy.__dict__['_tf_extension_type_is_packed'] = value
    return copy

  def _to_legacy_output_shapes(self):
    # [FUNCTION DOC]
    # Supports the data input pipeline's legacy shape protocol by returning this specification's shape
    # attribute. Raises a not-implemented error with advice to add a shape property if there is none.
    """Returns the shape property."""
    try:
      return self.shape
    except AttributeError as e:
      raise NotImplementedError(
          'It appears that the Spec of the ExtensionType is missing a shape'
          ' property. In order to support tf.Data, it is recommended that you'
          ' implement a shape property on the Spec.'
      ) from e


class _ExtensionTypeSpecCodec:
  # [CLASS DOC]
  # Codec registered with the structured value coder (see the registration just after this class) to
  # serialize and deserialize extension type specifications to and from structured value messages.
  # Stateless; implements the can-encode, encode, can-decode and decode steps of the codec protocol.
  """Codec for `tf.ExtensionTypeSpec`."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Input: any object. Returns true only if it is an extension type specification whose class is
    # registered under a name in the type specification registry; false otherwise (including when the
    # registry lookup raises a value error for an unregistered class).
    """Returns true if `pyobj` can be encoded as an ExtensionTypeSpec."""
    if isinstance(pyobj, ExtensionTypeSpec):
      try:
        type_spec_registry.get_name(type(pyobj))
        return True
      except ValueError:
        return False
    return False

  def do_encode(self, extension_type_spec_value, encode_fn):
    # [FUNCTION DOC]
    # Inputs: a registered extension type specification and a callback that encodes nested Python
    # state. Returns a structured value message holding a type specification record of the extension
    # type kind, with the registered class name, the encoded serialized state, and the count of flattened
    # component specifications. Raises a value error if the class is not registered.
    """Returns an encoded proto for the given `tf.ExtensionTypeSpec`."""
    type_spec_class_name = type_spec_registry.get_name(
        type(extension_type_spec_value)
    )

    type_state = extension_type_spec_value._serialize()  # pylint: disable=protected-access
    num_flat_components = len(
        nest.flatten(
            extension_type_spec_value._component_specs, expand_composites=True  # pylint: disable=protected-access
        )
    )
    encoded_type_spec = struct_pb2.StructuredValue()
    encoded_type_spec.type_spec_value.CopyFrom(
        struct_pb2.TypeSpecProto(
            type_spec_class=struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC,
            type_state=encode_fn(type_state),
            type_spec_class_name=type_spec_class_name,
            num_flat_components=num_flat_components,
        )
    )
    return encoded_type_spec

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Input: a structured value message. Returns true if it holds a type specification record whose kind
    # is the extension type specification kind, otherwise false.
    """Returns true if `value` can be decoded into a `tf.ExtensionTypeSpec`."""
    if value.HasField('type_spec_value'):
      type_spec_class_enum = value.type_spec_value.type_spec_class
      return (
          type_spec_class_enum == struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC
      )
    return False

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Inputs: a structured value message and a callback that decodes nested state. Looks up the class by
    # its recorded name in the registry; if not registered, emits a warning and falls back to the
    # anonymous extension type specification class. Returns the specification deserialized from the
    # decoded state. Side effect: possible warning.
    """Returns the `tf.TypeSpec` encoded by the proto `value`."""
    type_spec_proto = value.type_spec_value
    class_name = type_spec_proto.type_spec_class_name

    try:
      type_spec_class = type_spec_registry.lookup(class_name)
    except ValueError:
      type_spec_class = AnonymousExtensionTypeSpec
      warnings.warn(
          f"The type '{class_name}' has not been registered. "
          'Falling back to using AnonymousExtensionTypeSpec '
          'instead.'
      )

    # pylint: disable=protected-access
    return type_spec_class._deserialize(decode_fn(type_spec_proto.type_state))


nested_structure_coder.register_codec(_ExtensionTypeSpecCodec())


@tf_export('experimental.ExtensionTypeBatchEncoder')
class ExtensionTypeBatchEncoder(type_spec.TypeSpecBatchEncoder):
  # [CLASS DOC]
  # Default encoder that lets extension types be batched and unbatched by data pipelines, the neural
  # network library and mapped functions. Provides methods to compute the batched and unbatched
  # specification, to encode a value as a nest of tensors, to decode it back, and to describe the
  # encoding. Defaults assume every nested tensor, extension type, composite tensor and shape field
  # can be simply stacked or concatenated; types where that is false supply a custom subclass.
  # Stateless. Exported in the experimental API.
  """Class used to encode and decode extension type values for batching.

  In order to be batched and unbatched by APIs such as `tf.data.Dataset`,
  `tf.keras`, and `tf.map_fn`, extension type values must be encoded as a list
  of `tf.Tensor`s, where stacking, unstacking, or concatenating these encoded
  tensors and then decoding the result must be equivalent to stacking,
  unstacking, or concatenating the original values. `ExtensionTypeBatchEncoder`s
  are responsible for implementing this encoding.

  The default `ExtensionTypeBatchEncoder` that is used by
  `BatchableExtensionType` assumes that extension type values can be stacked,
  unstacked, or concatenated by simply stacking, unstacking, or concatenating
  every nested `Tensor`, `ExtensionType`, `CompositeTensor`, and `TensorShape`
  field.

  Extension types where this is not the case will need to override
  `__batch_encoder__` with a custom encoder that overrides the `batch`,
  `unbatch`, `encode`, and `decode` methods. E.g.:

  >>> class CustomBatchEncoder(ExtensionTypeBatchEncoder):
  ...   pass # Override batch(), unbatch(), encode(), and decode().

  >>> class CustomType(BatchableExtensionType):
  ...   x: tf.Tensor
  ...   y: tf.Tensor
  ...   shape: tf.TensorShape
  ...   __batch_encoder__ = CustomBatchEncoder()

  For example, `tf.RaggedTensor` and `tf.SparseTensor` both use custom batch
  encodings which define ops to "box" and "unbox" individual values into
  `tf.variant` tensors.
  """

  def batch(self, spec, batch_size):
    # [FUNCTION DOC]
    # Inputs: a specification for one value and the batch size (integer or null when unknown). Returns
    # a specification for a batch: a copy where every nested batchable specification is batched by its
    # own encoder and every shape field gets a new leading dimension of the batch size; other fields
    # are unchanged. Does not modify the input.
    """Returns the TypeSpec representing a batch of values described by `spec`.

    The default definition returns a `TypeSpec` that is equal to `spec`, except
    that an outer axis with size `batch_size` is added to every nested
    `TypeSpec` and `TensorShape` field.  Subclasses may override this default
    definition, when necessary.

    Args:
      spec: The `TypeSpec` for an individual value.
      batch_size: An `int` indicating the number of values that are batched
        together, or `None` if the batch size is not known.

    Returns:
      A `TypeSpec` for a batch of values.
    """

    def batch_field(f):
      # [FUNCTION DOC]
      # HELPER: nested function that batches one field of a specification. Batchable specifications
      # are batched with their own encoder, shapes get a leading batch dimension, anything else is returned
      # unchanged.
      if isinstance(f, type_spec.BatchableTypeSpec):
        return f.__batch_encoder__.batch(f, batch_size)
      elif isinstance(f, tensor_shape.TensorShape):
        return [batch_size] + f
      else:
        return f

    fields = tuple(spec.__dict__.items())
    batched_fields = nest.map_structure(batch_field, fields)
    return _create_object_from_type_and_dict(type(spec), batched_fields)

  def unbatch(self, spec):
    # [FUNCTION DOC]
    # Input: a specification for a batch. Returns the specification of one element: a copy where nested
    # batchable specifications are unbatched by their own encoder and shape fields lose their first
    # dimension; other fields are unchanged. Does not modify the input.
    """Returns the TypeSpec for a single unbatched element in `spec`.

    The default definition returns a `TypeSpec` that is equal to `spec`, except
    that the outermost axis is removed from every nested `TypeSpec`, and
    `TensorShape` field.  Subclasses may override this default definition, when
    necessary.

    Args:
      spec: The `TypeSpec` for a batch of values.

    Returns:
      A `TypeSpec` for an individual value.
    """

    def unbatch_field(f):
      # [FUNCTION DOC]
      # HELPER: nested function that unbatches one field of a specification. Batchable specifications use
      # their own encoder, shapes drop the first dimension, anything else is returned unchanged.
      if isinstance(f, type_spec.BatchableTypeSpec):
        return f.__batch_encoder__.unbatch(f)
      elif isinstance(f, tensor_shape.TensorShape):
        return f[1:]
      else:
        return f

    fields = tuple(spec.__dict__.items())
    unbatched_fields = nest.map_structure(unbatch_field, fields)
    return _create_object_from_type_and_dict(type(spec), unbatched_fields)

  def encode(self, spec, value, minimum_rank=0):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the minimum rank argument is documented but ignored by this default implementation.
    # Inputs: the specification, a value compatible with it, and the minimum rank (default zero).
    # Returns the components of the value as given by the specification (a tuple of tensors and
    # composite tensors in depth-first field order). Raises nothing itself.
    """Encodes `value` as a nest of batchable Tensors or CompositeTensors.

    The default definition returns a flat tuple of all the `Tensor`s,
    `CompositeTensor`s, and `ExtensionType`s from a depth-first traversal of
    `value`'s fields. Subclasses may override this default definition, when
    necessary.

    Args:
      spec: The TypeSpec of the value to encode.
      value: A value compatible with `spec`.
      minimum_rank: The minimum rank for the returned Tensors, CompositeTensors,
        and ExtensionType values.  This can be used to ensure that the encoded
        values can be unbatched this number of times.   If `minimum_rank>0`,
        then `t.shape[:minimum_rank]` must be compatible for all values `t`
        returned by `encode`.

    Returns:
      A nest (as defined by `tf.nest`) of `tf.Tensor`s, batchable
      `tf.CompositeTensor`s, or `tf.ExtensionType`s.  Stacking, unstacking, or
      concatenating these encoded values and then decoding the result must be
      equivalent to stacking, unstacking, or concatenating the original values.
    """
    return spec._to_components(value)  # pylint: disable=protected-access

  def decode(self, spec, encoded_value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the result is compatible with a name that is not a parameter (it
    # means the spec argument).
    # Inputs: the specification of the result and the encoded nest (from encoding, possibly stacked,
    # unstacked or concatenated). Returns the value rebuilt from components by the specification.
    """Decodes `value` from a batchable tensor encoding.

    See `encode` for a description of the default encoding.  Subclasses may
    override this default definition, when necessary.

    Args:
      spec: The TypeSpec for the result value.  If encoded values with spec `s`
        were batched, then `spec` should be `s.batch(batch_size)`; or if encoded
        values with spec `s` were unbatched, then `spec` should be
        `s.unbatch()`.
      encoded_value: A nest of values returned by `encode`; or a nest of values
        that was formed by stacking, unstacking, or concatenating the
        corresponding elements of values returned by `encode`.

    Returns:
      A value compatible with `type_spec`.
    """
    return spec._from_components(encoded_value)  # pylint: disable=protected-access

  def encoding_specs(self, spec):
    # [FUNCTION DOC]
    # KNOWN ISSUE: documented as returning a list of tensor specifications, but it returns whatever the
    # component specifications property yields, which is a tuple or a single specification.
    # Input: a specification. Returns the specifications of the components that encoding produces.
    """Returns a list of `TensorSpec`(s) describing the encoding for `spec`.

    See `encode` for a description of the default encoding.  Subclasses may
    override this default definition, when necessary.

    Args:
      spec: The TypeSpec whose encoding should be described.

    Returns:
      A nest (as defined by `tf.nest) of `tf.TypeSpec`, describing the values
      that are returned by `self.encode(spec, ...)`.  All TypeSpecs in this
      nest must be batchable.
    """
    return spec._component_specs  # pylint: disable=protected-access


class BatchableExtensionTypeSpec(
    ExtensionTypeSpec, type_spec.BatchableTypeSpec
):
  # [CLASS DOC]
  # Specification base class for batchable extension types. Combines the extension type specification
  # with the batchable type specification protocol, delegating batching and unbatching to the batch
  # encoder stored in a class attribute (the default encoder unless the type supplies its own), and
  # converting values to and from flat lists of tensors using the shared batchable helpers.
  """Base class for TypeSpecs for BatchableExtensionTypes."""

  __batch_encoder__ = ExtensionTypeBatchEncoder()

  def _batch(self, batch_size):
    # [FUNCTION DOC]
    # Input: batch size. Returns the batched specification, computed by the batch encoder.
    return self.__batch_encoder__.batch(self, batch_size)

  def _unbatch(self):
    # [FUNCTION DOC]
    # Returns the specification of a single element, computed by the batch encoder.
    return self.__batch_encoder__.unbatch(self)

  def _to_tensor_list(self, value):
    # [FUNCTION DOC]
    # Input: a value. Returns a flat list of tensors encoding it, using the shared batchable helper.
    return type_spec.batchable_to_tensor_list(self, value)

  def _to_batched_tensor_list(self, value):
    # [FUNCTION DOC]
    # Input: a value that is already batched. Returns a flat list of tensors encoding it, requiring a
    # minimum rank of one so the batch dimension can be unstacked.
    return type_spec.batchable_to_tensor_list(self, value, minimum_rank=1)

  def _from_compatible_tensor_list(self, tensor_list):
    # [FUNCTION DOC]
    # Input: a flat list of tensors. Returns the value rebuilt from them using the shared batchable helper.
    return type_spec.batchable_from_tensor_list(self, tensor_list)

  @property
  def _flat_tensor_specs(self):
    # [FUNCTION DOC]
    # Read-only property: the flat list of tensor specifications describing the encoded form, computed
    # by the shared batchable helper.
    return type_spec.get_batchable_flat_tensor_specs(self)


@tf_export('experimental.BatchableExtensionType')
class BatchableExtensionType(ExtensionType):
  # [CLASS DOC]
  # Public base class for extension types that can be batched and unbatched (usable with data
  # pipelines, the neural network library and mapped functions). Behaves like the plain extension type
  # base but its generated specification derives from the batchable specification class; a custom
  # batch encoder may be supplied through a class attribute. Carries the marker that stops the
  # metaclass from transforming this framework class.
  """An ExtensionType that can be batched and unbatched.

  `BatchableExtensionType`s can be used with APIs that require batching or
  unbatching, including `Keras`, `tf.data.Dataset`, and `tf.map_fn`.  E.g.:

  >>> class Vehicle(tf.experimental.BatchableExtensionType):
  ...   top_speed: tf.Tensor
  ...   mpg: tf.Tensor
  >>> batch = Vehicle([120, 150, 80], [30, 40, 12])
  >>> tf.map_fn(lambda vehicle: vehicle.top_speed * vehicle.mpg, batch,
  ...           fn_output_signature=tf.int32).numpy()
  array([3600, 6000,  960], dtype=int32)

  An `ExtensionTypeBatchEncoder` is used by these APIs to encode `ExtensionType`
  values. The default encoder assumes that values can be stacked, unstacked, or
  concatenated by simply stacking, unstacking, or concatenating every nested
  `Tensor`, `ExtensionType`, `CompositeTensor`, or `TensorShape` field.
  Extension types where this is not the case will need to override
  `__batch_encoder__` with a custom `ExtensionTypeBatchEncoder`.  See
  `tf.experimental.ExtensionTypeBatchEncoder` for more details.
  """

  # Let the metaclass know that it should *not* transform this class (since
  # this class is part of the ExtensionType framework, and not a user class).
  _tf_extension_type_do_not_transform_this_class = True


# For Pickle __reduce__ protocol:
def _deserialize_for_reduce(value_type, serialization):
  # [FUNCTION DOC]
  # HELPER: pickle reconstruction function for specifications. Inputs: the extension type class and
  # a serialized state. Returns the specification instance rebuilt through the class's nested
  # specification type.
  return value_type.Spec._deserialize(serialization)  # pylint: disable=protected-access


def _replace_tensor_with_spec(value):
  # [FUNCTION DOC]
  # HELPER: used when building a specification from a value. Input: any field value. Returns a tensor
  # specification (shape and type, name left out) for a tensor, the type specification for anything that
  # has one, and the value itself otherwise.
  if isinstance(value, tensor.Tensor):
    # Note: we intentionally exclude `value.name` from the `TensorSpec`.
    return tensor.TensorSpec(value.shape, value.dtype)
  if hasattr(value, '_type_spec'):
    return value._type_spec  # pylint: disable=protected-access
  return value


def _change_nested_mappings_to(value, new_type):
  # [FUNCTION DOC]
  # HELPER: converts between mutable and immutable mappings during specification serialization.
  # Inputs: a value and a mapping constructor. Recursively converts every mapping (plain or immutable)
  # into the new mapping type, descending into tuples; other values are returned unchanged. Pure.
  """Recursively replace mappings with `new_type`."""
  if isinstance(value, (dict, immutable_dict.ImmutableDict)):
    return new_type(
        [
            (k, _change_nested_mappings_to(v, new_type))
            for (k, v) in value.items()
        ]
    )
  elif isinstance(value, tuple):
    return tuple(_change_nested_mappings_to(elt, new_type) for elt in value)
  else:
    return value


# ==============================================================================
# Helper methods for tf.ExtensionTypeMetaclass
# ==============================================================================


def _check_field_annotations(cls):
  # [FUNCTION DOC]
  # HELPER: used by the metaclass to validate a new extension type class. Input: the class. Raises a
  # value error when a nested specification attribute is not a class or does not derive directly from the
  # base type specification, when any class attribute or annotation uses a reserved name, or when a
  # class attribute that is not callable, a property, a class or static method, a special name or an
  # internal marker lacks a type annotation. Returns nothing.
  """Validates the field annotations for tf.ExtensionType subclass `cls`."""
  annotations = getattr(cls, '__annotations__', {})

  # Check that no fields use reserved names.
  for name, value in cls.__dict__.items():
    if name == 'Spec':
      if not isinstance(value, type):
        raise ValueError(
            f'{cls.__qualname__}.Spec must be a nested class; got {value}.'
        )
      if value.__bases__ != (type_spec.TypeSpec,) and value.__bases__ != (
          object,
      ):
        raise ValueError(
            f'{cls.__qualname__}.Spec must be directly subclassed '
            'from tf.TypeSpec.'
        )
    elif extension_type_field.ExtensionTypeField.is_reserved_name(name):
      raise ValueError(
          f'The field annotations for {cls.__name__} are '
          f"invalid. Field '{name}' is reserved."
      )
  for name in annotations:
    if extension_type_field.ExtensionTypeField.is_reserved_name(name):
      raise ValueError(
          f'The field annotations for {cls.__name__} are '
          f"invalid. Field '{name}' is reserved."
      )

  # Check that all fields have type annotaitons.
  for key, value in cls.__dict__.items():
    if not (
        key in annotations
        or callable(value)
        or key.startswith('_abc_')
        or key == '_tf_extension_type_fields'
        or key.startswith('__')
        and key.endswith('__')
        or isinstance(value, (property, classmethod, staticmethod))
    ):
      raise ValueError(
          f'The field annotations for {cls.__name__} are '
          f'invalid. Field {key} is missing a type annotation.'
      )


def _add_extension_type_constructor(cls):
  # [FUNCTION DOC]
  # HELPER: used by the metaclass. Input: a class. If the class defines its own constructor, wraps it so
  # that fields may be set and then converted and validated; otherwise generates a constructor from
  # the declared fields. Replaces the constructor of the class.
  """Creates a constructor for a ExtensionType or ExtensionTypeSpec subclass."""
  if '__init__' in cls.__dict__:
    _wrap_user_constructor(cls)
  else:
    _build_extension_type_constructor(cls)


def _wrap_user_constructor(cls):
  # [FUNCTION DOC]
  # HELPER: wraps user-defined constructors of extension types and specifications.
  # KNOWN ISSUE: the wrapper removes the in-constructor marker without a finally clause, so if the user
  # code raises, the marker stays on the half-built instance.
  # Used for user-defined constructors. Input: a class. Replaces its constructor with a wrapper
  # preserving the original signature and metadata, which sets the in-constructor marker, runs the user
  # code, removes the marker, converts fields to canonical form, and runs the validation hook.
  """Wraps a user-defined constructor for tf.ExtensionType subclass `cls`."""
  user_constructor = cls.__init__

  def wrapped_init(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested wrapper that replaces a user constructor. Takes any positional and keyword
    # arguments, marks the instance as inside its constructor, forwards the arguments to the user
    # constructor, clears the marker, converts the fields, and runs validation. Returns nothing.
    self.__dict__[_IN_CONSTRUCTOR] = True
    user_constructor(self, *args, **kwargs)
    del self.__dict__[_IN_CONSTRUCTOR]

    self._tf_extension_type_convert_fields()  # pylint: disable=protected-access
    self.__validate__()

  cls.__init__ = tf_decorator.make_decorator(user_constructor, wrapped_init)


_NO_DEFAULT = extension_type_field.ExtensionTypeField.NO_DEFAULT


def _build_extension_type_constructor(cls):
  # [FUNCTION DOC]
  # HELPER: generates a constructor for an extension type class that has none. Input: the class.
  # Builds a signature from the field list: fields before the first defaulted field are
  # positional-or-keyword; once a defaulted field has been seen, a later field with no default and every
  # field after it become keyword-only; each parameter carries the field default and annotation.
  # The generated constructor binds its arguments to that signature (type error on mismatch),
  # applies defaults, stores the values, converts fields and runs validation. A signature is attached
  # for inspection tools and the constructor is installed on the class.
  """Builds a constructor for tf.ExtensionType subclass `cls`."""
  fields = cls._tf_extension_type_fields()  # pylint: disable=protected-access

  # Mark any no-default fields that follow default fields as keyword_only.
  got_default = False
  keyword_only_start = len(fields)
  for i in range(len(fields)):
    if got_default:
      if fields[i].default is _NO_DEFAULT:
        keyword_only_start = i
        break
    elif fields[i].default is not _NO_DEFAULT:
      got_default = True

  params = []
  for i, field in enumerate(fields):
    if i < keyword_only_start:
      kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
    else:
      kind = tf_inspect.Parameter.KEYWORD_ONLY
    if field.default is _NO_DEFAULT:
      default = tf_inspect.Parameter.empty
    else:
      default = field.default
    params.append(
        tf_inspect.Parameter(
            field.name, kind, default=default, annotation=field.value_type
        )
    )

  signature = tf_inspect.Signature(params, return_annotation=cls.__name__)

  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: nested generated constructor of an extension type. Binds the positional and keyword
    # arguments to the field signature (raising a type error when they do not match), fills in
    # defaults, stores all values on the instance, converts fields to canonical form, and runs the
    # validation hook. Returns nothing.
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    self.__dict__.update(bound_args.arguments)
    self._tf_extension_type_convert_fields()  # pylint: disable=protected-access
    self.__validate__()

  # __signature__ is supported by some inspection/documentation tools
  # (but note: typing.get_type_hints does not respect __signature__).
  __init__.__signature__ = tf_inspect.Signature(
      [tf_inspect.Parameter('self', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD)]
      + params,
      return_annotation=cls,
  )

  cls.__init__ = __init__


def _build_spec_constructor(cls):
  # [FUNCTION DOC]
  # HELPER: generates a constructor for a specification class that has none. Input: the specification
  # class. Builds a signature with one positional-or-keyword parameter per field (no defaults),
  # attaches it for inspection tools, and installs a constructor on the class.
  """Builds a constructor for ExtensionTypeSpec subclass `cls`."""
  params = []
  kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
  for field in cls._tf_extension_type_fields():  # pylint: disable=protected-access
    params.append(tf_inspect.Parameter(field.name, kind))

  signature = tf_inspect.Signature(params, return_annotation=cls.__name__)

  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: nested generated constructor of a specification. Binds the arguments to the field
    # signature (type error on mismatch), fills defaults, stores the values, converts fields for
    # specifications, and runs the validation hook. Returns nothing.
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    self.__dict__.update(bound_args.arguments)
    self._tf_extension_type_convert_fields()  # pylint: disable=protected-access
    self.__validate__()

  # __signature__ is supported by some inspection/documentation tools.
  __init__.__signature__ = tf_inspect.Signature(
      [tf_inspect.Parameter('self', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD)]
      + params,
      return_annotation=cls,
  )

  cls.__init__ = __init__


def _add_type_spec(cls):
  # [FUNCTION DOC]
  # HELPER: used by the metaclass to create the nested specification class. Input: the extension type
  # class. Copies the user-supplied nested specification attributes (skipping module, dictionary and weak
  # reference entries), raising a value error if one uses a reserved name or shadows a field. Chooses
  # the batchable specification base for batchable types (inheriting a batch encoder if the user
  # specification lacks one) or the plain base otherwise, raising a value error if a batch encoder is
  # defined on a non-batchable type. Creates the specification class with the proper module and
  # qualified name, attaches it to the class, installs its constructor, removes the type specification
  # property from the abstract method set, and, if the class defines a name, registers the
  # specification under that name plus a suffix in the registry.
  """Creates a nested TypeSpec class for tf.ExtensionType subclass `cls`."""
  spec_name = cls.__name__ + '.Spec'
  spec_qualname = cls.__qualname__ + '.Spec'

  # Set __module__ explicitly as a dynamic created class has module='abc'
  # by default.
  spec_dict = {'value_type': cls, '__module__': cls.__module__}

  # Copy user-supplied customizations into the TypeSpec.
  user_spec = cls.__dict__.get('Spec', None)
  if user_spec is not None:
    for name, value in user_spec.__dict__.items():
      if extension_type_field.ExtensionTypeField.is_reserved_name(name):
        raise ValueError(
            f"TypeSpec {spec_qualname} uses reserved name '{name}'."
        )
      if cls._tf_extension_type_has_field(name):  # pylint: disable=protected-access
        raise ValueError(
            f"TypeSpec {spec_qualname} defines a variable '{name}'"
            f' which shadows a field in {cls.__qualname__}'
        )
      if name in ('__module__', '__dict__', '__weakref__'):
        continue

      spec_dict[name] = value

  if issubclass(cls, BatchableExtensionType):
    type_spec_base = BatchableExtensionTypeSpec
    if (
        hasattr(cls, '__batch_encoder__')
        and '__batch_encoder__' not in spec_dict
    ):
      spec_dict['__batch_encoder__'] = cls.__batch_encoder__
  else:
    type_spec_base = ExtensionTypeSpec
    if hasattr(cls, '__batch_encoder__') or '__batch_encoder__' in spec_dict:
      raise ValueError(
          '__batch_encoder__ should only be defined for '
          'BatchableExtensionType classes.'
      )

  # Build the TypeSpec and store it as a nested class inside `cls`.
  spec = type(spec_name, (type_spec_base,), spec_dict)
  spec.__qualname__ = spec_qualname
  setattr(cls, 'Spec', spec)

  # Build a constructor for the TypeSpec class.
  if '__init__' in spec.__dict__:
    _wrap_user_constructor(spec)
  else:
    _build_spec_constructor(spec)

  cls.__abstractmethods__ -= {'_type_spec'}

  # If the user included an explicit `__name__` attribute, then use that to
  # register the TypeSpec (so it can be used in SavedModel signatures).
  if '__name__' in cls.__dict__:
    type_spec_registry.register(cls.__dict__['__name__'] + '.Spec')(spec)


# ==============================================================================
# Anonymous ExtensionType
# ==============================================================================
class AnonymousExtensionType(ExtensionType):
  # [CLASS DOC]
  # Fallback extension type used when a saved model's signature mentions an extension type that was
  # not registered. Fields are arbitrary keyword arguments restricted to simple supported kinds
  # (numbers, strings, bytes, null, dtypes, shapes, tensors, composites, and tuples or mappings of
  # these). Instances are fully immutable and convertible to or from concrete types via reinterpret.
  # Carries the marker that stops the metaclass from transforming it.
  """Fallback used to decode `tf.ExtensionType` when the original type is unavailable.

  When a SavedModel is serialized, the signatures of any functions in the
  SavedModel can include `tf.ExtensionType` subclasses.  These subclasses are
  usually
  registered, so they can be restored when the SavedModel is loaded.  However,
  if a SavedModel is loaded without first registering the ExtensionType types in
  its
  signature, then the SavedModel will fall back to using the
  `AnonymousExtensionType`
  type instead.

  If necessary, `AnonymousExtensionType` objects can be converted to a concrete
  `tf.ExtensionType` subclass (and vice versa) using `reinterpret`.
  """

  # Let the metaclass know that it should *not* transform this class (since
  # this class is part of the ExtensionType framework, and not a user class).
  _tf_extension_type_do_not_transform_this_class = True

  def __init__(self, **fields):
    # [FUNCTION DOC]
    # Constructor. Takes arbitrary named field values. Raises a value error if any name is reserved
    # or is a special double-underscore name. Each value is validated and converted (unsupported
    # kinds raise a value error; mappings become immutable mappings), stored directly in the instance,
    # then fields are converted again and the base constructor is called. Returns nothing.
    for name in fields:
      if extension_type_field.ExtensionTypeField.is_reserved_name(name) or (
          name.startswith('__') and name.endswith('__')
      ):
        raise ValueError(
            f'Reserved field name {name} was encountered '
            'when trying to instantiate an AnonymousExtensionType.'
        )
    fields = [(k, _convert_anonymous_fields(v)) for (k, v) in fields.items()]
    self.__dict__.update(fields)
    self._tf_extension_type_convert_fields()
    super().__init__()

  @classmethod
  def _tf_extension_type_fields(cls):
    # [FUNCTION DOC]
    # KNOWN ISSUE: builds the field list from the class dictionary rather than from an instance, so it
    # describes class attributes and not the fields given to a particular instance.
    # Class method returning untyped field descriptions, one for every non-reserved name in the
    # class dictionary.
    return [
        extension_type_field.ExtensionTypeField(name, None)
        for name in cls.__dict__
        if not extension_type_field.ExtensionTypeField.is_reserved_name(name)
    ]

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Always raises an attribute error: anonymous extension type instances cannot be modified.
    raise AttributeError(
        f'Cannot set attribute `{name}`. '
        'AnonymousExtensionType instances are immutable.'
    )

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Always raises an attribute error: anonymous extension type instances cannot be modified.
    raise AttributeError(
        f'Cannot delete attribute `{name}`. '
        'AnonymousExtensionType instances are immutable.'
    )

  def _tf_extension_type_convert_fields(self):
    # [FUNCTION DOC]
    # Re-validates and converts every non-reserved stored field value in place using the anonymous
    # conversion helper (raising a value error for unsupported kinds). Returns nothing.
    fields = [
        (k, _convert_anonymous_fields(v))
        for (k, v) in self.__dict__.items()
        if not extension_type_field.ExtensionTypeField.is_reserved_name(k)
    ]
    self.__dict__.update(fields)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string made of the fixed anonymous type name and all non-reserved fields as name and value
    # pairs separated by commas.
    fields = [
        f'{k}={v!r}'
        for (k, v) in self.__dict__.items()
        if not extension_type_field.ExtensionTypeField.is_reserved_name(k)
    ]
    return f'AnonymousExtensionType({", ".join(fields)})'

  _tf_extension_type_cached_type_spec = None

  @property
  def _type_spec(self):  # CompositeTensor API.
    # [FUNCTION DOC]
    # Read-only property: the anonymous specification for this value, built on first use and cached in the
    # instance dictionary.
    # Note: the TypeSpec contains all static (non-tensor) data from `self`.
    if self._tf_extension_type_cached_type_spec is None:
      spec = AnonymousExtensionTypeSpec.from_value(self)
      self.__dict__['_tf_extension_type_cached_type_spec'] = spec
    return self._tf_extension_type_cached_type_spec


@type_spec_registry.register('tf.AnonymousExtensionType.Spec')
class AnonymousExtensionTypeSpec(ExtensionTypeSpec):
  # [CLASS DOC]
  # Specification for anonymous extension types, registered in the type specification registry under
  # a fixed name so that saved models can restore it. Holds its fields directly in the instance
  # dictionary, treats them as immutable, and reports the anonymous extension type as its value type.
  """TypeSpec for AnonymousExtensionType."""

  def __init__(self, **fields):
    # [FUNCTION DOC]
    # Constructor. Takes arbitrary named field values. Raises a value error if any name is reserved or is
    # a special double-underscore name. Values are converted in specification mode (tensors are not
    # accepted but type specifications are; mappings become immutable), stored directly, then the base
    # constructor is called. Returns nothing.
    for name in fields:
      if extension_type_field.ExtensionTypeField.is_reserved_name(name) or (
          name.startswith('__') and name.endswith('__')
      ):
        raise ValueError(
            f'Reserved field name {name} was encountered '
            'when trying to instantiate an AnonymousExtensionTypeSpec.'
        )
    fields = [
        (k, _convert_anonymous_fields(v, for_spec=True))
        for (k, v) in fields.items()
    ]
    self.__dict__.update(fields)
    super().__init__()

  value_type = AnonymousExtensionType  # TypeSpec API.

  def _serialize(self):  # TypeSpec API.
    # [FUNCTION DOC]
    # Returns a tuple of name and value pairs for all non-reserved fields, with nested immutable mappings
    # converted to plain dictionaries. (The packed flag is not included, unlike the base class.)
    return tuple(
        (name, _change_nested_mappings_to(value, dict))
        for (name, value) in self.__dict__.items()
        if not extension_type_field.ExtensionTypeField.is_reserved_name(name)
    )

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Allows only the cached fixed properties of type specifications to be set (delegating to the
    # parent); any other name raises an attribute error because the specification is immutable.
    if name in type_spec.CACHED_FIXED_PROPERTIES:
      super().__setattr__(name, value)
    else:
      raise AttributeError(
          f'Cannot set attribute `{name}`. '
          'AnonymousExtensionTypeSpec instances are immutable.'
      )

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Always raises an attribute error: instances are immutable.
    raise AttributeError(
        f'Cannot delete attribute `{name}`. '
        'AnonymousExtensionTypeSpec instances are immutable.'
    )


def _convert_anonymous_fields(value, for_spec=False):
  # [FUNCTION DOC]
  # HELPER: validates and normalizes field values of anonymous extension types and their specifications.
  # Inputs: a value and a flag saying the value is for a specification (default false). Simple values
  # (integers, floats, booleans, strings, bytes, null, dtypes, shapes) pass through; tuples and
  # mappings are converted recursively (mappings become immutable); tensors and composite tensors are
  # accepted only when not for a specification; type specifications only when for one. Anything else
  # raises a value error.
  """Type-checks and converts `value` for inclusion in an AnonymousExtensionType."""
  if isinstance(
      value,
      (
          int,
          float,
          bool,
          str,
          bytes,
          type(None),
          dtypes.DType,
          tensor_shape.TensorShape,
      ),
  ):
    return value

  if isinstance(value, tuple):
    return tuple(_convert_anonymous_fields(v, for_spec) for v in value)

  if isinstance(value, typing.Mapping):
    return immutable_dict.ImmutableDict(
        [
            (
                _convert_anonymous_fields(k, for_spec),
                _convert_anonymous_fields(v, for_spec),
            )
            for (k, v) in value.items()
        ]
    )

  if (
      isinstance(value, (tensor.Tensor, composite_tensor.CompositeTensor))
      and not for_spec
  ):
    return value

  if isinstance(value, type_spec.TypeSpec) and for_spec:
    return value

  raise ValueError(
      'Cannot convert anonymous fields from '
      f'an unsupported `value` argument: {value!r}.'
  )


# ==============================================================================
# reinterpret
# ==============================================================================
def reinterpret(value, new_type):
  # [FUNCTION DOC]
  # Converts an extension type value to another extension type with compatible fields; for example
  # a concrete type to the anonymous type or the reverse. Inputs: an extension type instance and a
  # subclass of the extension type base. Raises a value error if either argument is of the wrong kind.
  # Copies all non-reserved fields into a new object of the new type without running its constructor,
  # converts the fields (type-checking them against the new annotations) and runs the new type's
  # validation hook. Returns the new instance.
  """Converts a given `ExtensionType` to a new type with compatible fields.

  In particular, this can be used to convert a concrete subclass of
  `ExtensionType` to an `AnonymousExtensionType`, or vice versa.  When
  converting to a non-anonymous ExtensionType, field values are type-checked to
  ensure they are consistent with `new_type`'s type annotations, and validated
  with `new_type.__validate__`.

  Args:
    value: An instance of a subclass of `tf.ExtensionType`
    new_type: A subclass of `tf.ExtensionType`

  Returns:
    An instance of `new_type`, whose fields are copied from `value`.
  """
  if not isinstance(value, ExtensionType):
    raise ValueError(
        'reinterpret expects `value` to be a tf.ExtensionType instance; '
        f'got {value!r}'
    )
  if not (isinstance(new_type, type) and issubclass(new_type, ExtensionType)):
    raise ValueError(
        'reinterpret expects `new_type` to be a subclass of tf.ExtensionType; '
        f'got {new_type!r}'
    )

  fields = [
      item
      for item in value.__dict__.items()
      if not extension_type_field.ExtensionTypeField.is_reserved_name(item[0])
  ]
  new_value = _create_object_from_type_and_dict(new_type, fields)
  new_value._tf_extension_type_convert_fields()  # pylint: disable=protected-access
  new_value.__validate__()
  return new_value
