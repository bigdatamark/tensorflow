# [MODULE DOC]
# Core AutoGraph conversion policy helpers. Exposes is_unsupported (entities AutoGraph never
# handles), is_allowlisted (entities that are called as-is based on configured conversion rules,
# generator status, call overrides, owner classes and namedtuples), and a small cache API
# (is_in_allowlist_cache, cache_allowlisted) backed by a module-level unbound instance cache.
# Used by the api module to decide whether to convert a callable. Import-time side effect:
# creates the global allowlist cache.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Core conversion logic, serves as main point of access."""

import functools
import inspect
import sys
import unittest

from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.util import tf_inspect


_ALLOWLIST_CACHE = cache.UnboundInstanceCache()


def _is_of_known_loaded_module(f, module_name):
  # [FUNCTION DOC]
  # HELPER: used by is_unsupported to detect members of selected standard modules.
  # Returns true if the named module is already loaded and f is (by identity) one of the values in
  # that module's namespace. Inputs: f (entity), module_name (string). Returns false if the module
  # is not loaded or f is not found. No side effects.
  mod = sys.modules.get(module_name, None)
  if mod is None:
    return False
  if any(v is not None for v in mod.__dict__.values() if f is v):
    return True
  return False


def _is_known_loaded_type(f, module_name, entity_name):
  # [FUNCTION DOC]
  # HELPER: type check used by is_unsupported to detect wrapt and lru_cache wrappers without
  # importing those modules.
  # Returns true if the named module is already loaded and has the named type, and f is an
  # instance of that type or is a bound method whose underlying function is such an instance.
  # Inputs: f (entity), module_name (string), entity_name (string). Returns false otherwise,
  # including when the module is not loaded. No side effects.
  """Tests whether the function or method is an instance of a known type."""
  if (module_name not in sys.modules or
      not hasattr(sys.modules[module_name], entity_name)):
    return False
  type_entity = getattr(sys.modules[module_name], entity_name)
  if isinstance(f, type_entity):
    # The method if of this type. Example:
    #
    # o = ClassType()
    # function(o.method)()
    return True
  # Note: inspect is required here, to avoid unpacking tf.function decorators.
  if inspect.ismethod(f):
    # The unbound method if of this type. Example:
    #
    # class ClassType:
    #   @function
    #   def method(self):
    #     ...
    # o = ClassType()
    # o.method()
    if isinstance(f.__func__, type_entity):
      return True
  return False


def is_unsupported(o):
  # [FUNCTION DOC]
  # Decides whether AutoGraph cannot or should not handle an entity at all, so it runs as-is.
  # Input: o (any entity). Returns true for entities decorated by the wrapt library (also emits a
  # warning), functions wrapped by the least-recently-used cache decorator, constructors, members
  # of a few standard modules (collections, pdb, copy, inspect, re), and custom-op plugin
  # entities; false otherwise. Side effect: logging.
  """Checks whether an entity is supported by AutoGraph at all."""

  # TODO(b/122265385): Remove this bypass.
  if (_is_known_loaded_type(o, 'wrapt', 'FunctionWrapper') or
      _is_known_loaded_type(o, 'wrapt', 'BoundFunctionWrapper')):
    logging.warning(
        '{} appears to be decorated by wrapt, which is not yet supported'
        ' by AutoGraph. The function will run as-is.'
        ' You may still apply AutoGraph before the wrapt decorator.'.format(o))
    logging.log(2, 'Permanently allowed: %s: wrapt decorated', o)
    return True

  if _is_known_loaded_type(o, 'functools', '_lru_cache_wrapper'):
    logging.log(2, 'Permanently allowed: %s: lru_cache', o)
    return True

  # Constructors are permanently allowed.
  # TODO(mdan): Toggle as experimental feature instead.
  # TODO(b/124016764): Remove this limitation.
  if inspect_utils.isconstructor(o):
    logging.log(2, 'Permanently allowed: %s: constructor', o)
    return True

  # Other built-in modules are permanently allowed.
  # TODO(mdan): Figure out how to do this consistently for all stdlib modules.
  if any(
      _is_of_known_loaded_module(o, m)
      for m in ('collections', 'pdb', 'copy', 'inspect', 're')):
    logging.log(2, 'Permanently allowed: %s: part of builtin module', o)
    return True

  # Custom ops and kernels are also permanently allowed.
  # See tensorflow.framework.load_library.
  if (hasattr(o, '__module__') and
      hasattr(o.__module__, '_IS_TENSORFLOW_PLUGIN')):
    logging.log(2, 'Permanently allowed: %s: TensorFlow plugin', o)
    return True

  return False


# TODO(mdan): allow_namedtuple_subclass should be hardcoded to True.
def is_allowlisted(
    o, check_call_override=True, allow_namedtuple_subclass=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says allow_namedtuple_subclass, when true, means namedtuple
  # subclasses are not allowed; the code actually uses it to allow only namedtuple types that do
  # not themselves derive from another namedtuple (direct namedtuple types), and when false it
  # allows any namedtuple or subclass.
  # Decides whether an entity should be treated as allowlisted, meaning AutoGraph will not convert
  # it and will call it as-is. Inputs: o (any Python entity); check_call_override (boolean,
  # default true; when true, a non-class object is allowed if its call method is allowed);
  # allow_namedtuple_subclass (boolean, default false, see above). Returns a boolean.
  # Rules, in order: the entity's module is matched against the configured conversion rules
  # (first rule giving convert means not allowed, do-not-convert means allowed; partial objects
  # are treated as belonging to the functools module); generator functions are allowed; callable
  # objects are allowed if their call method is; bound methods are allowed if the owner class is a
  # unit-test case subclass or the defining class is itself allowlisted (the method target
  # wrapper class is unwrapped to its target class first); namedtuple types are allowed per the
  # flag. Otherwise not allowed. Side effect: debug logging of the decision.
  """Checks whether an entity is allowed for use in graph mode.

  Examples of allowed entities include all members of the tensorflow
  package.

  Args:
    o: A Python entity.
    check_call_override: Reserved for internal use. When set to `False`, it
      disables the rule according to which classes are allowed if their
      __call__ method is allowed.
    allow_namedtuple_subclass: Reserved for internal use. When `True`,
      namedtuple subclasses are not allowed.

  Returns:
    Boolean
  """
  # TODO(b/120224672): Fix this.
  if isinstance(o, functools.partial):
    # tf_inspect.getmodule(functools.partial(...)) otherwise returns None since
    # functools.partial objects do not have a __module__ attribute.
    m = functools
  else:
    m = tf_inspect.getmodule(o)

  # Examples of callables that lack a __module__ property include builtins.
  if hasattr(m, '__name__'):
    for rule in config.CONVERSION_RULES:
      action = rule.get_action(m)
      if action == config.Action.CONVERT:
        logging.log(2, 'Not allowed: %s: %s', o, rule)
        return False
      elif action == config.Action.DO_NOT_CONVERT:
        logging.log(2, 'Allowlisted: %s: %s', o, rule)
        return True

  # The check for __code__ below is because isgeneratorfunction crashes
  # without one.
  if hasattr(o, '__code__') and tf_inspect.isgeneratorfunction(o):
    logging.log(2, 'Allowlisted: %s: generator functions are not converted', o)
    return True

  if (check_call_override and not tf_inspect.isclass(o) and
      hasattr(o, '__call__')):
    # Callable objects: allowed if their __call__ method is.
    # The type check avoids infinite recursion around the __call__ method
    # of function objects.
    if (type(o) != type(o.__call__)) and is_allowlisted(o.__call__):  # pylint: disable=unidiomatic-typecheck
      logging.log(2, 'Allowlisted: %s: object __call__ allowed', o)
      return True

  owner_class = None
  if tf_inspect.ismethod(o):
    # Methods of allowed classes are also allowed, even if they are
    # bound via user subclasses.
    #
    # For example, suppose `tf.Foo` has a method called `bar`, and `baz` is
    # defined as below. `tf.Foo` is allowed. Then `baz.bar` is also
    # allowed.
    #
    #   class Custom(tf.Foo):
    #     pass
    #
    #   baz = Custom()
    #
    # For the example above, if `Custom` did overload `bar`, then it would no
    # longer be allowed.

    owner_class = inspect_utils.getmethodclass(o)
    if owner_class is tf_method_target.TfMethodTarget:
      owner_class = o.__self__.target_class
    if owner_class is not None:
      if issubclass(owner_class, unittest.TestCase):
        logging.log(2, 'Allowlisted: %s: method of TestCase subclass', o)
        return True

      owner_class = inspect_utils.getdefiningclass(o, owner_class)
      if is_allowlisted(
          owner_class,
          check_call_override=False,
          allow_namedtuple_subclass=True):
        logging.log(2, 'Allowlisted: %s: owner is allowed %s', o,
                    owner_class)
        return True

  if inspect_utils.isnamedtuple(o):
    # Due to the way they're constructed, namedtuple types cannot be converted
    # because they don't expose source code. But we assume they are safe for
    # graph mode since they are just containers.
    if allow_namedtuple_subclass:
      if not any(inspect_utils.isnamedtuple(base) for base in o.__bases__):
        logging.log(2, 'Allowlisted: %s: named tuple', o)
        return True
    else:
      logging.log(2, 'Allowlisted: %s: named tuple or subclass', o)
      return True

  logging.log(2, 'Not allowed: %s: default rule', o)
  return False


def is_in_allowlist_cache(entity, options):
  # [FUNCTION DOC]
  # Looks up whether the entity was previously recorded as allowlisted for the given options in
  # the module-level allowlist cache. Inputs: entity (any object) and options (the conversion
  # options used as sub-key). Returns a boolean; returns false when the entity is unhashable or
  # cannot be weakly referenced (the type error is swallowed). No side effects.
  try:
    return _ALLOWLIST_CACHE.has(entity, options)
  except TypeError:
    # Catch-all for entities that are unhashable or don't allow weakrefs.
    return False


def cache_allowlisted(entity, options):
  # [FUNCTION DOC]
  # Records in the module-level allowlist cache that the given entity is allowlisted under the
  # given conversion options. Inputs: entity (any object, ideally hashable and weak-referenceable)
  # and options (the options object used as the cache sub-key). Returns nothing. Side effect:
  # mutates the global allowlist cache. If the entity is unhashable or cannot be weakly
  # referenced, the resulting type error is swallowed and nothing is cached.
  try:
    _ALLOWLIST_CACHE[entity][options] = True
  except TypeError:
    # Catch-all for entities that are unhashable or don't allow weakrefs.
    pass
