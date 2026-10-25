# [MODULE DOC]
# KNOWN ISSUE: the Python 2 branch (the else branch of the interpreter-version check) is dead code
# because it is only reached when the major version is not 3; only the Python 3 tables are used.
# Provides PythonObjectToProtoVisitor, a visitor that is called for every module/class found while
# walking the public TensorFlow Python API, and records a description of each one as an API-object
# protocol buffer (module with members, class with base classes and members, or message class with
# its descriptor). The collected protos are used to generate and compare golden API files.
# Also holds module-private tables that normalize Python-version-specific type names, a table of
# special-case objects that are skipped or hard-coded, and helpers that produce address-free
# argument-spec strings, parse pybind11-style docstring signatures and trim method resolution
# orders. Import-time side effect: builds the normalization tables, depending on the interpreter
# version.
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
#
# ==============================================================================
"""A visitor class that generates protobufs for each python object."""

import enum
import re
import sys

from google.protobuf import message
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.api.lib import api_objects_pb2

# Following object need to be handled individually.
_CORNER_CASES = {
    '': {
        'tools': {}
    },
    'test.TestCase': {},
    'test.TestCase.failureException': {},
    'train.NanLossDuringTrainingError': {
        'message': {}
    },
    'estimator.NanLossDuringTrainingError': {
        'message': {}
    },
    'train.LooperThread': {
        'isAlive': {},
        'join': {},
        'native_id': {}
    }
}

# Python 2 vs. 3 differences
if sys.version_info.major == 3:
  _NORMALIZE_TYPE = {}
  for t in ('property', 'object', 'getset_descriptor', 'int', 'str', 'type',
            'tuple', 'module', 'collections.defaultdict', 'set', 'dict',
            'NoneType', 'frozenset', 'member_descriptor'):
    _NORMALIZE_TYPE["<class '%s'>" % t] = "<type '%s'>" % t
  for e in 'Exception', 'RuntimeError':
    _NORMALIZE_TYPE["<class '%s'>" % e] = "<type 'exceptions.%s'>" % e
  _NORMALIZE_TYPE["<class 'abc.ABCMeta'>"] = "<type 'type'>"
  _NORMALIZE_ISINSTANCE = {
      "<class "
      "'tensorflow.lite.python.op_hint.OpHint.OpHintArgumentTracker'>":  # pylint: disable=line-too-long
          "<class "
          "'tensorflow.lite.python.op_hint.OpHintArgumentTracker'>",
      "<class "
      "'tensorflow.python.training.monitored_session._MonitoredSession.StepContext'>":  # pylint: disable=line-too-long
          "<class "
          "'tensorflow.python.training.monitored_session.StepContext'>",
      "<class "
      "'tensorflow.python.ops.variables.Variable.SaveSliceInfo'>":
          '<class '
          "'tensorflow.python.ops.variables.SaveSliceInfo'>"
  }

  def _SkipMember(cls, member):
    # [FUNCTION DOC]
    # HELPER: filter used while adding members to a proto (Python 3 variant).
    # Inputs: cls, the object whose members are being listed; member, a member name. Returns true
    # when the member should be left out: the name of the exception-chaining method, or the name or
    # value attribute when cls is an enumeration class. Returns false otherwise. No side effects.
    return (member == 'with_traceback' or member in ('name', 'value') and
            isinstance(cls, type) and issubclass(cls, enum.Enum))
else:
  _NORMALIZE_TYPE = {
      "<class 'abc.ABCMeta'>": "<type 'type'>",
      "<class 'pybind11_type'>": "<class 'pybind11_builtins.pybind11_type'>",
  }
  _NORMALIZE_ISINSTANCE = {
      "<class 'pybind11_object'>":
          "<class 'pybind11_builtins.pybind11_object'>",
  }

  def _SkipMember(cls, member):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # HELPER: member filter (Python 2 variant).
    # KNOWN ISSUE: dead code, only reached if the interpreter major version is not 3.
    # Inputs: cls and member are ignored. Always returns
    # false, so no member is skipped.
    return False


# Differences created by typing implementations.
_NORMALIZE_TYPE[
    'tensorflow.python.framework.tensor.Tensor'
] = "<class 'tensorflow.python.framework.tensor.Tensor'>"
_NORMALIZE_TYPE['typing.Generic'] = "<class 'typing.Generic'>"
# TODO(b/203104448): Remove once the golden files are generated in Python 3.7.
_NORMALIZE_TYPE["<class 'typing._GenericAlias'>"] = 'typing.Union'
# TODO(b/203104448): Remove once the golden files are generated in Python 3.9.
_NORMALIZE_TYPE["<class 'typing._UnionGenericAlias'>"] = 'typing.Union'
# TODO(b/203104448): Remove once the golden files are generated in Python 3.8.
_NORMALIZE_TYPE["<class 'typing_extensions._ProtocolMeta'>"] = (
    '<class '
    "'typing._ProtocolMeta'>")
# TODO(b/203104448): Remove once the golden files are generated in Python 3.8.
_NORMALIZE_TYPE[
    "<class 'typing_extensions.Protocol'>"] = "<class 'typing.Protocol'>"

if sys.version_info.major == 3 and sys.version_info.minor >= 8:
  _NORMALIZE_TYPE["<class '_collections._tuplegetter'>"] = "<type 'property'>"


def _NormalizeType(ty):
  # [FUNCTION DOC]
  # HELPER: maps a type description string to its canonical golden-file spelling.
  # Input: ty, a string such as the text form of a type. Returns the replacement from the type
  # normalization table when present, otherwise ty unchanged. No side effects.
  return _NORMALIZE_TYPE.get(ty, ty)


def _NormalizeIsInstance(ty):
  # [FUNCTION DOC]
  # HELPER: maps a class description string used in an is-instance list to its canonical spelling.
  # Input: ty, a string. Returns the replacement from the is-instance normalization table when
  # present (for example, nested classes reported under their public names), else ty unchanged.
  return _NORMALIZE_ISINSTANCE.get(ty, ty)


def _SanitizedArgSpec(obj):
  # [FUNCTION DOC]
  # HELPER: builds a stable text form of a routine's argument specification for golden files.
  # KNOWN ISSUE: the docstring says it returns a sanitized list of base classes and calls the
  # parameter something to create an argspec of; it actually returns one string describing the
  # argument names, variadic and keyword argument names, and defaults of a routine.
  # Input: obj, a Python function or method. Output: a string of the form listing the argument
  # names, the variadic-argument name, the keyword-argument name, and the defaults (as a list of
  # strings, or the word None when there are none, in which case the text has no trailing comma).
  # Defaults whose text contains a memory address are shortened to the part before the address
  # followed by an instance marker so output is reproducible. May propagate errors from the
  # argument inspection when obj is not introspectable.
  """Get an ArgSpec string that is free of addresses.

  We have callables as function arg defaults. This results in addresses in
  getargspec output. This function returns a sanitized string list of base
  classes.

  Args:
    obj: A python routine for us the create the sanitized arspec of.

  Returns:
    string, a string representation of the argspec.
  """
  output_string = ''
  unsanitized_arg_spec = tf_inspect.getargspec(obj)

  for clean_attr in ('args', 'varargs', 'keywords'):
    output_string += '%s=%s, ' % (clean_attr,
                                  getattr(unsanitized_arg_spec, clean_attr))

  if unsanitized_arg_spec.defaults:
    sanitized_defaults = []
    for val in unsanitized_arg_spec.defaults:
      str_val = str(val)
      # Sanitize argspecs that have hex code in them.
      if ' at 0x' in str_val:
        sanitized_defaults.append('%s instance>' % str_val.split(' at ')[0])
      else:
        sanitized_defaults.append(str_val)

    output_string += 'defaults=%s, ' % sanitized_defaults

  else:
    output_string += 'defaults=None'

  return output_string


def _GenerateArgsSpec(doc):
  # [FUNCTION DOC]
  # HELPER: extracts quoted argument names from a signature at the start of a docstring.
  # Input: doc, a docstring text. Finds the first parenthesized group, splits it on commas and
  # takes the first word of each part as an argument name. Returns the names, each in single
  # quotes, joined by comma and space; returns nothing/null if there is no parenthesized group or
  # any part has no word characters.
  # KNOWN ISSUE: an empty parameter list yields a single empty part, so zero-argument signatures
  # return null; commas inside nested brackets or default values are also split incorrectly.
  """Generate args spec from a method docstring."""
  args_spec = []
  doc = re.search(r'\(.*\)', doc)
  if not doc:
    return None
  # remove parentheses
  doc = doc.group().strip('(').strip(')')
  doc_split = doc.split(',')
  for s in doc_split:
    arg = re.search(r'\w+', s)
    if not arg:
      return None
    args_spec.append(f'\'{arg.group()}\'')
  return ', '.join(args_spec)


def _ParseDocstringArgSpec(doc):
  # [FUNCTION DOC]
  # HELPER: builds an argument-spec string for native-extension functions from their docstring.
  # KNOWN ISSUE: the docstring says null is returned on failure, but the function raises a value
  # error instead; it also raises a type error (not handled by its caller) if doc is null.
  # Input: doc, a docstring that should begin with a function signature in pybind11 style.
  # Output: a string naming the argument names with no variadic, keyword or default information.
  # Raises a value error when the docstring does not start with a signature or the names cannot
  # be extracted. No side effects.
  """Get an ArgSpec string from a method docstring.

  This method is used to generate argspec for C extension functions that follow
  pybind11 DocString format function signature. For example:
  `foo_function(a: int, b: string) -> None...`

  Args:
    doc: A python string which starts with function signature.

  Returns:
    string: a argspec string representation if successful. If not, return None.

  Raises:
    ValueError: Raised when failed to parse the input docstring.
  """
  # Check if the docstring begins with a function signature
  match = re.search(r'^\w+\(.*\)', doc)
  args_spec = _GenerateArgsSpec(doc)
  if (not match) or (args_spec is None):
    raise ValueError(f'Failed to parse argspec from docstring: {doc}')

  # TODO(panzf): implement parsing docs with varargs, keywords, and defaults
  output_string = (
      f'args=[{args_spec}], varargs=None, keywords=None, defaults=None')
  return output_string


def _SanitizedMRO(obj):
  # [FUNCTION DOC]
  # HELPER: computes a version-robust list of base class names for a class.
  # KNOWN ISSUE: the docstring describes obj as a routine whose argspec is wanted; it is actually
  # a class whose method resolution order is listed.
  # Input: obj, a class. Walks its method resolution order, skipping classes created by the
  # deprecated-alias decorator, appending the normalized text of each class, and stopping after
  # the first class that is neither a TensorFlow nor a Keras class, or after the test stub-out
  # class. Returns the list of strings. No side effects.
  """Get a list of superclasses with minimal amount of non-TF classes.

  Based on many parameters like python version, OS, protobuf implementation
  or changes in google core libraries the list of superclasses of a class
  can change. We only return the first non-TF class to be robust to non API
  affecting changes. The Method Resolution Order returned by `tf_inspect.getmro`
  is still maintained in the return value.

  Args:
    obj: A python routine for us the create the sanitized arspec of.

  Returns:
    list of strings, string representation of the class names.
  """
  return_list = []
  for cls in tf_inspect.getmro(obj):
    if cls.__name__ == '_NewClass':
      # Ignore class created by @deprecated_alias decorator.
      continue
    str_repr = _NormalizeType(str(cls))
    return_list.append(str_repr)
    # Class type that has keras in their name should also be monitored. This
    # will cover any class imported from third_party/py/keras.
    if 'tensorflow' not in str_repr and 'keras' not in str_repr:
      break

    # Hack - tensorflow.test.StubOutForTesting may or may not be type <object>
    # depending on the environment. To avoid inconsistency, break after we add
    # StubOutForTesting to the return_list.
    if 'StubOutForTesting' in str_repr:
      break

  return return_list


def _IsProtoClass(obj):
  # [FUNCTION DOC]
  # HELPER: tells whether an object is a protocol buffer message class. Input: obj, any object.
  # Returns true when it is a class deriving from the protobuf message base class, else false.
  """Returns whether the passed obj is a Protocol Buffer class."""
  return isinstance(obj, type) and issubclass(obj, message.Message)


class PythonObjectToProtoVisitor:
  # [CLASS DOC]
  # Visitor object for the API traversal utility: instances are called once per visited module or
  # class with its path, the object and its children, and accumulate one API-object proto per
  # visited path. State: a mapping from library path (with the default root prefix) to proto, and
  # the root prefix (default is the TensorFlow package name). Retrieve results with GetProtos.
  """A visitor that summarizes given python objects as protobufs."""

  def __init__(self, default_path='tensorflow'):
    # [FUNCTION DOC]
    # Constructor. Input: default_path, the root path prefix prepended to every visited object's
    # path (default is the TensorFlow package name). Creates an empty mapping of protos keyed by
    # path. Returns nothing.
    # A dict to store all protocol buffers.
    # Keyed by "path" to the object.
    self._protos = {}
    self._default_path = default_path

  def GetProtos(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a list, but it returns the mapping from path to
    # API-object proto, and it is the live internal mapping (not a copy).
    # No inputs. Returns everything recorded so far by visits.
    """Return the list of protos stored."""
    return self._protos

  def __call__(self, path, parent, children):
    # [FUNCTION DOC]
    # Visit callback. Inputs: path, dotted path of the visited object relative to the root (may be
    # empty for the root); parent, the visited module or class (unwrapped from decorators);
    # children, a list of (name, object) pairs. Builds the full library path, then, unless the path
    # is a fully-skipped special case, records into the protos mapping: a module proto with its
    # public members, a message-class proto with the message descriptor, or a class proto with its
    # normalized base classes and members. Special-case tables can drop or hard-code members.
    # Anything else logs an error and records nothing. Returns nothing; mutates the mapping.
    # The path to the object.
    lib_path = self._default_path + '.' + path if path else self._default_path
    _, parent = tf_decorator.unwrap(parent)

    # A small helper method to construct members(children) protos.
    def _AddMember(member_name, member_obj, proto):
      # [FUNCTION DOC]
      # HELPER: adds one child to the proto under construction (nested helper of the visit callback).
      # Inputs: member_name, the child name; member_obj, the child object (decorators are
      # unwrapped); proto, the module or class proto to extend. Skips members rejected by the
      # version-specific filter and hidden-API attributes. Only public names and the constructor
      # are kept. Routines get a method entry with an argument spec (from code inspection, or parsed
      # from the docstring for native functions, left empty on parse failure); other objects get a
      # member entry with a normalized type text (modules get a fixed module type text). Returns
      # nothing; mutates proto.
      """Add the child object to the object being constructed."""
      _, member_obj = tf_decorator.unwrap(member_obj)
      if (_SkipMember(parent, member_name) or
          isinstance(member_obj, deprecation.HiddenTfApiAttribute)):
        return
      if member_name == '__init__' or not member_name.startswith('_'):
        if tf_inspect.isroutine(member_obj):
          new_method = proto.member_method.add()
          new_method.name = member_name
          # If member_obj is a python builtin, there is no way to get its
          # argspec, because it is implemented on the C side. It also has no
          # func_code.
          if hasattr(member_obj, '__code__'):
            new_method.argspec = _SanitizedArgSpec(member_obj)
          else:
            # Try to parse argspec based on docstring for exposed C++ functions
            if member_name != '__init__' and hasattr(member_obj, '__doc__'):
              doc = member_obj.__doc__
              try:
                spec_str = _ParseDocstringArgSpec(doc)
              except ValueError:
                pass
              else:
                new_method.argspec = spec_str
        else:
          new_member = proto.member.add()
          new_member.name = member_name
          if tf_inspect.ismodule(member_obj):
            new_member.mtype = "<type \'module\'>"
          else:
            new_member.mtype = _NormalizeType(str(type(member_obj)))

    parent_corner_cases = _CORNER_CASES.get(path, {})

    if path not in _CORNER_CASES or parent_corner_cases:
      # Decide if we have a module or a class.
      if tf_inspect.ismodule(parent):
        # Create a module object.
        module_obj = api_objects_pb2.TFAPIModule()
        for name, child in children:
          if name in parent_corner_cases:
            # If we have an empty entry, skip this object.
            if parent_corner_cases[name]:
              module_obj.member.add(**(parent_corner_cases[name]))
          else:
            _AddMember(name, child, module_obj)

        # Store the constructed module object.
        self._protos[lib_path] = api_objects_pb2.TFAPIObject(
            path=lib_path, tf_module=module_obj)
      elif _IsProtoClass(parent):
        proto_obj = api_objects_pb2.TFAPIProto()
        parent.DESCRIPTOR.CopyToProto(proto_obj.descriptor)

        # Store the constructed proto object.
        self._protos[lib_path] = api_objects_pb2.TFAPIObject(
            path=lib_path, tf_proto=proto_obj)
      elif tf_inspect.isclass(parent):
        # Construct a class.
        class_obj = api_objects_pb2.TFAPIClass()
        class_obj.is_instance.extend(
            _NormalizeIsInstance(i) for i in _SanitizedMRO(parent))
        for name, child in children:
          if name in parent_corner_cases:
            # If we have an empty entry, skip this object.
            if parent_corner_cases[name]:
              class_obj.member.add(**(parent_corner_cases[name]))
          else:
            _AddMember(name, child, class_obj)

        # Store the constructed class object.
        self._protos[lib_path] = api_objects_pb2.TFAPIObject(
            path=lib_path, tf_class=class_obj)
      else:
        logging.error(
            'Illegal call to ApiProtoDump::_py_obj_to_proto.'
            'Object is neither a module nor a class: %s', path)
