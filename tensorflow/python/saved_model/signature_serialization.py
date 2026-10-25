# [MODULE DOC]
# Helpers used by the SavedModel save and load code to turn user-supplied functions into
# serving signatures. Exposes functions to pick a default function to export, to canonicalize a
# user "signatures" argument into concrete functions that return mappings of named tensors, to
# wrap signatures in an immutable trackable signature-map object, and to validate that an object
# graph does not misuse the reserved signatures attribute. Defines the default-signature
# attribute name and the signatures attribute name constants. Import-time side effect: registers
# the "signature_map" revived type so loading recreates signature map objects.
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
"""Helpers for working with signatures in tf.saved_model.save."""

from absl import logging

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc

DEFAULT_SIGNATURE_ATTR = "_default_save_signature"
SIGNATURE_ATTRIBUTE_NAME = "signatures"
# Max number of warnings to show if signature contains normalized input names.
_NUM_DISPLAY_NORMALIZED_SIGNATURES = 5


def _get_signature(function):
  # [FUNCTION DOC]
  # HELPER: Normalizes a user function for the signature export logic.
  # Takes a polymorphic function or concrete function. If it is a polymorphic function that
  # has an input signature, returns its (garbage-collected) concrete function; if the result is a
  # concrete function, returns it, otherwise returns nothing. Has no other side effects.
  if (
      isinstance(function, def_function.Function)
      and function.input_signature is not None
  ):
    function = function._get_concrete_function_garbage_collected()  # pylint: disable=protected-access
  if not isinstance(function, defun.ConcreteFunction):
    return None
  return function


def _valid_signature(concrete_function):
  # [FUNCTION DOC]
  # HELPER: Predicate used when choosing a function to export as a signature.
  # Returns false if the concrete function has no outputs, or if input validation or output
  # normalization raises a value error (for example variable inputs or non-tensor outputs);
  # otherwise returns true. Catches only value errors and has no side effects.
  """Returns whether concrete function can be converted to a signature."""
  if not concrete_function.outputs:
    # Functions without outputs don't make sense as signatures. We just don't
    # have any way to run an Operation with no outputs as a SignatureDef in the
    # 1.x style.
    return False
  try:
    _validate_inputs(concrete_function)
    _normalize_outputs(concrete_function.structured_outputs, "unused", "unused")
  except ValueError:
    return False
  return True


def _validate_inputs(concrete_function):
  # [FUNCTION DOC]
  # HELPER: Input check for signature export.
  # Scans the flattened structured input signature of a concrete function and raises a value
  # error if any input is a variable spec, since such functions cannot be exported as signatures.
  # Returns nothing otherwise.
  """Raises error if input type is tf.Variable."""
  if any(
      isinstance(inp, resource_variable_ops.VariableSpec)
      for inp in nest.flatten(concrete_function.structured_input_signature)
  ):
    raise ValueError(
        f"Unable to serialize concrete_function '{concrete_function.name}'"
        "with tf.Variable input. Functions that expect tf.Variable "
        "inputs cannot be exported as signatures."
    )


def _get_signature_name_changes(concrete_function):
  # [FUNCTION DOC]
  # HELPER: Supports the informational logging in canonicalizing signatures.
  # Pairs the function definition's signature input arguments with the graph inputs and returns
  # a mapping from each user-specified input name to the normalized name, only for inputs where
  # the two differ. Inputs without a user-specified-name attribute (value error) are skipped.
  # No side effects.
  """Checks for user-specified signature input names that are normalized."""
  # Map of {user-given name: normalized name} if the names are un-identical.
  name_changes = {}
  for signature_input_name, graph_input in zip(
      concrete_function.function_def.signature.input_arg,
      concrete_function.graph.inputs,
  ):
    try:
      user_specified_name = compat.as_str(
          graph_input.op.get_attr("_user_specified_name")
      )
      if signature_input_name.name != user_specified_name:
        name_changes[user_specified_name] = signature_input_name.name
    except ValueError:
      # Signature input does not have a user-specified name.
      pass
  return name_changes


def find_function_to_export(saveable_view):
  # [FUNCTION DOC]
  # Chooses a function to export as the default serving signature when the user gave none.
  # Takes a saveable view of the object graph and inspects the children of its root. If a child
  # function is named with the default-signature attribute name it is returned immediately.
  # Otherwise, if exactly one child function yields a valid signature, that concrete function is
  # returned; if none or several qualify, returns nothing. Does not modify anything.
  """Function to export, None if no suitable function was found."""
  # If the user did not specify signatures, check the root object for a function
  # that can be made into a signature.
  children = saveable_view.list_children(saveable_view.root)

  # TODO(b/205014194): Discuss removing this behaviour. It can lead to WTFs when
  # a user decides to annotate more functions with tf.function and suddenly
  # serving that model way later in the process stops working.
  possible_signatures = []
  for name, child in children:
    if not isinstance(child, (def_function.Function, defun.ConcreteFunction)):
      continue
    if name == DEFAULT_SIGNATURE_ATTR:
      return child
    concrete = _get_signature(child)
    if concrete is not None and _valid_signature(concrete):
      possible_signatures.append(concrete)

  if len(possible_signatures) == 1:
    single_function = possible_signatures[0]
    signature = _get_signature(single_function)
    if signature and _valid_signature(signature):
      return signature
  return None


def canonicalize_signatures(signatures):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring says it returns a dictionary of concrete functions, but it
  # actually returns a three-tuple: concrete signatures, wrapped functions and default tensors.
  # Converts the user "signatures" argument into signature concrete functions. Input is nothing,
  # a mapping from signature keys to functions, or a single function (stored under the default
  # serving key). Nothing yields three empty mappings. Each function must be a polymorphic
  # function with an input signature or a concrete function, else a value error is raised;
  # variable inputs also raise a value error. Cached variables are wrapped, a wrapper returning
  # a normalized mapping of output names to tensors is traced with named tensor specs, allowed
  # function attributes are propagated, and positional-argument count is set to one when there
  # is a single input and zero otherwise. Returns the mapping of keys to final concrete
  # functions, the mapping of original to wrapped functions, and a mapping of keys to
  # argument-name to default tensor mappings. Side effect: logs up to a few informational
  # messages about input names that get renamed, and sets private attributes on concrete functions.
  """Converts `signatures` into a dictionary of concrete functions."""
  if signatures is None:
    return {}, {}, {}
  if not isinstance(signatures, collections_abc.Mapping):
    signatures = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signatures
    }
  num_normalized_signatures_counter = 0
  concrete_signatures = {}
  wrapped_functions = {}
  defaults = {}
  for signature_key, function in signatures.items():
    original_function = signature_function = _get_signature(function)
    if signature_function is None:
      raise ValueError(
          "Expected a TensorFlow function for which to generate a signature, "
          f"but got {function}. Only `tf.functions` with an input signature or "
          "concrete functions can be used as a signature."
      )

    wrapped_functions[original_function] = signature_function = (
        wrapped_functions.get(original_function)
        or function_serialization.wrap_cached_variables(original_function)
    )
    _validate_inputs(signature_function)
    if num_normalized_signatures_counter < _NUM_DISPLAY_NORMALIZED_SIGNATURES:
      signature_name_changes = _get_signature_name_changes(signature_function)
      if signature_name_changes:
        num_normalized_signatures_counter += 1
        logging.info(
            "Function `%s` contains input name(s) %s with unsupported "
            "characters which will be renamed to %s in the SavedModel.",
            compat.as_str(signature_function.graph.name),
            ", ".join(signature_name_changes.keys()),
            ", ".join(signature_name_changes.values()),
        )

    # Re-wrap the function so that it returns a dictionary of Tensors. This
    # matches the format of 1.x-style signatures.
    # pylint: disable=cell-var-from-loop
    def signature_wrapper(**kwargs):
      # [FUNCTION DOC]
      # HELPER: Wrapper traced to form each exported signature function.
      # Takes keyword arguments (named tensors), calls the captured signature function with them
      # and returns the outputs normalized to a mapping of string names to tensors. Raises a value
      # error via normalization for bad outputs. Captures loop variables by reference.
      structured_outputs = signature_function(**kwargs)
      return _normalize_outputs(
          structured_outputs, signature_function.name, signature_key
      )

    if hasattr(function, "__name__"):
      signature_wrapper.__name__ = "signature_wrapper_" + function.__name__

    # Extract experimental attributes and propagate it to the wrapped_function.
    # At the moment only the `disable_summaries_at_runtime` attr needs to be
    # propagated.
    experimental_attributes = {}
    for attr in attributes.POLYMORPHIC_FUNCTION_ALLOWLIST:
      attr_value = signature_function.function_def.attr.get(attr, None)
      if attr != attributes.NO_INLINE and attr_value is not None:
        experimental_attributes[attr] = attr_value
    if not experimental_attributes:
      experimental_attributes = None

    wrapped_function = def_function.function(
        signature_wrapper, experimental_attributes=experimental_attributes
    )
    tensor_spec_signature = {}
    if signature_function.structured_input_signature is not None:
      # The structured input signature may contain other non-tensor arguments.
      inputs = filter(
          lambda x: isinstance(x, tensor.TensorSpec),
          nest.flatten(
              signature_function.structured_input_signature,
              expand_composites=True,
          ),
      )
    else:
      # Structured input signature isn't always defined for some functions.
      inputs = signature_function.inputs

    for keyword, inp in zip(
        signature_function._arg_keywords,  # pylint: disable=protected-access
        inputs,
    ):
      keyword = compat.as_str(keyword)
      if isinstance(inp, tensor.TensorSpec):
        spec = tensor.TensorSpec(inp.shape, inp.dtype, name=keyword)
      else:
        spec = tensor.TensorSpec.from_tensor(inp, name=keyword)
      tensor_spec_signature[keyword] = spec
    final_concrete = wrapped_function._get_concrete_function_garbage_collected(  # pylint: disable=protected-access
        **tensor_spec_signature
    )
    # pylint: disable=protected-access
    if len(final_concrete._arg_keywords) == 1:
      # If there is only one input to the signature, a very common case, then
      # ordering is unambiguous and we can let people pass a positional
      # argument. Since SignatureDefs are unordered (protobuf "map") multiple
      # arguments means we need to be keyword-only.
      final_concrete._num_positional_args = 1
    else:
      final_concrete._num_positional_args = 0
    # pylint: enable=protected-access
    concrete_signatures[signature_key] = final_concrete
    # pylint: enable=cell-var-from-loop
    if isinstance(function, core.PolymorphicFunction):
      flattened_defaults = nest.flatten(
          function.function_spec.fullargspec.defaults  # pylint: disable=protected-access
      )
      len_default = len(flattened_defaults or [])
      arg_names = list(tensor_spec_signature.keys())
      if len_default > 0:
        # tensor_spec_signature uses the same nest.flatten() as
        # flattened_defaults.
        for arg, default in zip(
            arg_names[-len_default:],  # pylint: disable=protected-access
            flattened_defaults or [],
        ):
          if not isinstance(default, tensor.Tensor):
            continue
          defaults.setdefault(signature_key, {})[arg] = default
  return concrete_signatures, wrapped_functions, defaults


def _normalize_outputs(outputs, function_name, signature_key):
  # [FUNCTION DOC]
  # HELPER: Output normalization for signature wrappers and validity checks.
  # Takes function outputs plus the function name and signature key (used only in error
  # messages). Mappings pass through; named tuples become mappings of field names; a single value
  # is wrapped in a list; sequences become a mapping with keys output_0, output_1 and so on.
  # Raises a value error if a key is not a string or a value is not a tensor or composite tensor.
  # Returns the resulting mapping. No side effects.
  """Normalize outputs if necessary and check that they are tensors."""
  # Convert `outputs` to a dictionary (if it's not one already).
  if not isinstance(outputs, collections_abc.Mapping):
    # Check if `outputs` is a namedtuple.
    if hasattr(outputs, "_asdict"):
      outputs = outputs._asdict()
    else:
      if not isinstance(outputs, collections_abc.Sequence):
        outputs = [outputs]
      outputs = {
          "output_{}".format(output_index): output
          for output_index, output in enumerate(outputs)
      }

  # Check that the keys of `outputs` are strings and the values are Tensors.
  for key, value in outputs.items():
    if not isinstance(key, compat.bytes_or_text_types):
      raise ValueError(
          f"Got a dictionary with a non-string key {key!r} in the output of "
          f"the function {compat.as_str_any(function_name)} used to generate "
          f"the SavedModel signature {signature_key!r}."
      )
    if not isinstance(value, (tensor.Tensor, composite_tensor.CompositeTensor)):
      raise ValueError(
          f"Got a non-Tensor value {value!r} for key {key!r} in the output of "
          f"the function {compat.as_str_any(function_name)} used to generate "
          f"the SavedModel signature {signature_key!r}. "
          "Outputs for functions used as signatures must be a single Tensor, "
          "a sequence of Tensors, or a dictionary from string to Tensor."
      )
  return outputs


# _SignatureMap is immutable to ensure that users do not expect changes to be
# reflected in the SavedModel. Using public APIs, tf.saved_model.load() is the
# only way to create a _SignatureMap and there is no way to modify it. So we can
# safely ignore/overwrite ".signatures" attributes attached to objects being
# saved if they contain a _SignatureMap. A ".signatures" attribute containing
# any other type (e.g. a regular dict) will raise an exception asking the user
# to first "del obj.signatures" if they want it overwritten.
class _SignatureMap(collections_abc.Mapping, base.Trackable):
  # [CLASS DOC]
  # HELPER: Internal container used to expose signatures on saved and loaded objects.
  # A read-only mapping from signature names to concrete functions that is also trackable, so
  # the functions become dependencies in the SavedModel object graph. Holds a private dictionary
  # of signatures; entries can only be added through a private method used while restoring.
  # Only reports children when saving as a SavedModel. Registered as a revived type.
  """A collection of SavedModel signatures."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor taking no arguments; creates an empty private dictionary of signatures.
    self._signatures = {}

  def _add_signature(self, name, concrete_function):
    # [FUNCTION DOC]
    # HELPER: Private mutator used by the restore code (the revived-type setter).
    # Stores the concrete function under the given signature name, overwriting any existing
    # entry. Mutates the map; returns nothing.
    """Adds a signature to the _SignatureMap."""
    # Ideally this object would be immutable, but restore is streaming so we do
    # need a private API for adding new signatures to an existing object.
    self._signatures[name] = concrete_function

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Returns the concrete function stored under the signature name; a missing name raises a
    # key error.
    return self._signatures[key]

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the signature names.
    return iter(self._signatures)

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of stored signatures.
    return len(self._signatures)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string showing the class name and the underlying signatures dictionary.
    return "_SignatureMap({})".format(self._signatures)

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    # [FUNCTION DOC]
    # Reports the trackable dependencies of this map. Takes the save type (default checkpoint)
    # and ignores extra keyword arguments. For anything other than a SavedModel save returns an
    # empty mapping; otherwise returns a mapping of name to function containing only the entries
    # that are polymorphic or concrete functions.
    if save_type != base.SaveType.SAVEDMODEL:
      return {}

    return {
        key: value
        for key, value in self.items()
        if isinstance(value, (def_function.Function, defun.ConcreteFunction))
    }


revived_types.register_revived_type(
    "signature_map",
    lambda obj: isinstance(obj, _SignatureMap),
    versions=[
        revived_types.VersionedTypeRegistration(
            # Standard dependencies are enough to reconstruct the trackable
            # items in dictionaries, so we don't need to save any extra
            # information.
            object_factory=lambda proto: _SignatureMap(),
            version=1,
            min_producer_version=1,
            min_consumer_version=1,
            setter=_SignatureMap._add_signature,  # pylint: disable=protected-access
        )
    ],
)


def create_signature_map(signatures):
  # [FUNCTION DOC]
  # Builds a signature-map object from a mapping of names to concrete functions (normally the
  # output of canonicalizing signatures). Asserts that each value is a concrete function with
  # mapping-valued structured outputs and that its positional-argument count is one for a single
  # input and zero otherwise; assertion failures raise assertion errors. Returns the new map.
  """Creates an object containing `signatures`."""
  signature_map = _SignatureMap()
  for name, func in signatures.items():
    # This true of any signature that came from canonicalize_signatures. Here as
    # a sanity check on saving; crashing on load (e.g. in _add_signature) would
    # be more problematic in case future export changes violated these
    # assertions.
    assert isinstance(func, defun.ConcreteFunction)
    assert isinstance(func.structured_outputs, collections_abc.Mapping)
    # pylint: disable=protected-access
    if len(func._arg_keywords) == 1:
      assert 1 == func._num_positional_args
    else:
      assert 0 == func._num_positional_args
    signature_map._add_signature(name, func)
    # pylint: enable=protected-access
  return signature_map


def validate_augmented_graph_view(augmented_graph_view):
  # [FUNCTION DOC]
  # Checks that the root object of an augmented graph view does not have a child named with the
  # reserved signatures attribute name unless it is a signature-map object. Otherwise raises a
  # value error telling the user to delete the attribute before saving. Returns nothing.
  """Performs signature-related sanity checks on `augmented_graph_view`."""
  for name, dep in augmented_graph_view.list_children(
      augmented_graph_view.root
  ):
    if name == SIGNATURE_ATTRIBUTE_NAME:
      if not isinstance(dep, _SignatureMap):
        raise ValueError(
            f"Exporting an object {augmented_graph_view.root} which has an"
            f" attribute named '{SIGNATURE_ATTRIBUTE_NAME}'. This is a reserved"
            " attribute used to store SavedModel signatures in objects which"
            " come from `tf.saved_model.load`. Delete this attribute (e.g."
            f" `del obj.{SIGNATURE_ATTRIBUTE_NAME}`) before saving if this"
            " shadowing is acceptable."
        )
      break
