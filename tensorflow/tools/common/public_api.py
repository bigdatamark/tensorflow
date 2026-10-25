# [MODULE DOC]
# HELPER: Support module for API-traversal tools (documentation generators and API golden checks).
# Defines a single visitor class that wraps another visitor so that traversal of the tensorflow
# package only reports the public API. Works together with the traverse module, which calls a
# visitor with a path, a parent object and a mutable list of children. There are no import-time
# side effects other than importing the regular-expression module and the tf inspection helpers.
# Exposes the class PublicAPIVisitor.
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
"""Visitor restricting traversal to only the public tensorflow API."""

import re

from tensorflow.python.util import tf_inspect


class PublicAPIVisitor:
  # [CLASS DOC]
  # HELPER: Filtering adapter used with the traverse function when walking the public TF API.
  # Wraps a user supplied visitor callable. Before the wrapped visitor sees the children of a
  # module or class, private members (underscore names, and entries in the private map) are
  # removed; after the wrapped visitor returns, members listed in the do-not-descend map are
  # removed so that traversal does not recurse into them. Holds the wrapped visitor, the root
  # name (default 'tf') used to build fully qualified paths, and two mappings from fully
  # qualified parent names to lists of member names. Instances are callable and are passed
  # directly to the traversal function.
  """Visitor to use with `traverse` to visit exactly the public TF API."""

  def __init__(self, visitor):
    # [FUNCTION DOC]
    # Constructor. Input: visitor, a callable taking a path, a parent object and a list of
    # (name, child) pairs; it is stored and invoked for public members only. Returns nothing.
    # Side effects: initializes the root name to 'tf', and builds the private map (names to
    # hide entirely, such as compiler, core, python, dtensor, security, tsl under 'tf' and
    # cpp_flags under 'tf.flags') and the do-not-descend map (names that are shown but not
    # explored, such as examples, flags, platform, tools, tensorboard under 'tf', flags under
    # 'tf.app', mock under 'tf.test'). Raises nothing.
    """Constructor.

    `visitor` should be a callable suitable as a visitor for `traverse`. It will
    be called only for members of the public TensorFlow API.

    Args:
      visitor: A visitor to call for the public API.
    """
    self._visitor = visitor
    self._root_name = 'tf'

    # Modules/classes we want to suppress entirely.
    self._private_map = {
        'tf': [
            'compiler',
            'core',
            # TODO(scottzhu): See b/227410870 for more details. Currently
            # dtensor API is exposed under tf.experimental.dtensor, but in the
            # meantime, we have tensorflow/dtensor directory which will be treat
            # as a python package. We want to avoid step into the
            # tensorflow/dtensor directory when visit the API.
            # When the tf.dtensor becomes the public API, it will actually pick
            # up from tf.compat.v2.dtensor as priority and hide the
            # tensorflow/dtensor package.
            'security',
            'dtensor',
            'python',
            'tsl',  # TODO(tlongeri): Remove after TSL is moved out of TF.
        ],
        # Some implementations have this internal module that we shouldn't
        # expose.
        'tf.flags': ['cpp_flags'],
    }

    # Modules/classes we do not want to descend into if we hit them. Usually,
    # system modules exposed through platforms for compatibility reasons.
    # Each entry maps a module path to a name to ignore in traversal.
    self._do_not_descend_map = {
        'tf': [
            'examples',
            'flags',  # Don't add flags
            # TODO(drpng): This can be removed once sealed off.
            'platform',
            # TODO(drpng): This can be removed once sealed.
            'pywrap_tensorflow',
            # TODO(drpng): This can be removed once sealed.
            'user_ops',
            'tools',
            'tensorboard',
        ],

        ## Everything below here is legitimate.
        # It'll stay, but it's not officially part of the API.
        'tf.app': ['flags'],
        # Imported for compatibility between py2/3.
        'tf.test': ['mock'],
    }

  @property
  def private_map(self):
    # [FUNCTION DOC]
    # Read-only property returning the live (not copied) mapping from fully qualified parent
    # names to lists of member names that are hidden completely. Callers may mutate it before
    # traversal starts. No inputs, no exceptions, no side effects.
    """A map from parents to symbols that should not be included at all.

    This map can be edited, but it should not be edited once traversal has
    begun.

    Returns:
      The map marking symbols to not include.
    """
    return self._private_map

  @property
  def do_not_descend_map(self):
    # [FUNCTION DOC]
    # Read-only property returning the live (not copied) mapping from fully qualified parent
    # names to lists of member names that are reported to the visitor but not explored further.
    # Callers may mutate it before traversal starts. No inputs, no exceptions, no side effects.
    """A map from parents to symbols that should not be descended into.

    This map can be edited, but it should not be edited once traversal has
    begun.

    Returns:
      The map marking symbols to not explore.
    """
    return self._do_not_descend_map

  def set_root_name(self, root_name):
    # [FUNCTION DOC]
    # Replaces the root name used as the first component of fully qualified paths when looking
    # up the private and do-not-descend maps. Input: root_name, a string. Returns nothing.
    # Side effect: mutates the visitor's root name. Note the maps themselves keep their
    # original keys starting with 'tf', so a different root name will not match them unless the
    # maps are edited too.
    """Override the default root name of 'tf'."""
    self._root_name = root_name

  def _is_private(self, path, name, obj=None):
    # [FUNCTION DOC]
    # HELPER: Member filter used by the call method of this visitor.
    # Decides whether a child should be hidden. Inputs: path, the fully qualified parent name
    # including the root name; name, the child's name; obj, the child object, which is ignored.
    # Returns true when the name is listed in the private map under the parent path, or when the
    # name starts with an underscore and is not a double-underscore dunder name, or when the
    # name is one of the base, class or next-in-method-resolution-order special attributes.
    # Otherwise returns false. No side effects and no exceptions.
    """Return whether a name is private."""
    # TODO(wicke): Find out what names to exclude.
    del obj  # Unused.
    return ((path in self._private_map and name in self._private_map[path]) or
            (name.startswith('_') and not re.match('__.*__$', name) or
             name in ['__base__', '__class__', '__next_in_mro__']))

  def _do_not_descend(self, path, name):
    # [FUNCTION DOC]
    # HELPER: Member filter used by the call method of this visitor.
    # Inputs: path, the fully qualified parent name; name, the child's name. Returns true if the
    # parent path has an entry in the do-not-descend map and the name is listed in it, else
    # false. Missing parent entries are handled safely. No side effects, no exceptions.
    """Safely queries if a specific fully qualified name should be excluded."""
    return (path in self._do_not_descend_map and
            name in self._do_not_descend_map[path])

  def __call__(self, path, parent, children):
    # [FUNCTION DOC]
    # Visitor entry point invoked by the traversal. Inputs: path, dotted name of the parent
    # relative to the root (empty for the root itself); parent, the module or class being
    # visited; children, a mutable list of (name, object) pairs. Steps: first raises a runtime
    # error if the parent is a module whose path has more than ten components, to fail fast on
    # accidental public imports; then builds the full path by prefixing the root name; removes
    # private children from the list in place; calls the wrapped visitor with the original
    # path, the parent and the filtered list; finally removes from the list, in place, the
    # children that must not be descended into. Returns nothing. Side effects: mutates the
    # children list and whatever the wrapped visitor does.
    """Visitor interface, see `traverse` for details."""

    # Avoid long waits in cases of pretty unambiguous failure.
    if tf_inspect.ismodule(parent) and len(path.split('.')) > 10:
      raise RuntimeError('Modules nested too deep:\n%s.%s\n\nThis is likely a '
                         'problem with an accidental public import.' %
                         (self._root_name, path))

    # Includes self._root_name
    full_path = '.'.join([self._root_name, path]) if path else self._root_name

    # Remove things that are not visible.
    for name, child in list(children):
      if self._is_private(full_path, name, child):
        children.remove((name, child))

    self._visitor(path, parent, children)

    # Remove things that are visible, but which should not be descended into.
    for name, child in list(children):
      if self._do_not_descend(full_path, name):
        children.remove((name, child))
