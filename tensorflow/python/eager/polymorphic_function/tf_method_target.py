# [MODULE DOC]
# HELPER: helper module defining the binding target used when methods are wrapped by the
# tracing function decorator. The target keeps references to the original instance and function
# so AutoGraph can recover the original self. No import-time side effects.
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
"""Module for the TFMethodTarget Class."""

import weakref

from tensorflow.python.util import tf_inspect


# When a method is bound to objects of this type, it allows AutoGraph to
# recover a weak reference the original method's self pointer, so that it can
# execute it consistent with class_method_to_instance_method's
# bound_method_wrapper.
# TODO(b/119246461): This is not pretty. Use a descriptor instead?
class TfMethodTarget:
  # [CLASS DOC]
  # Binding target used when a method is wrapped by the tracing function decorator. It keeps a
  # reference to the original instance (or class) and a weak reference to the original Python
  # function, letting AutoGraph recover the original self and call the method consistently
  # with the bound-method wrapper. Uses fixed slots for its two attributes.
  """Binding target for methods replaced by function and defun."""

  __slots__ = ("weakrefself_target__", "weakrefself_func__")

  def __init__(self, target, original_python_function):
    # [FUNCTION DOC]
    # Constructor. Inputs: target, which is stored as given and later called with no arguments
    # to obtain the real self (so it is expected to be a weak reference or similar callable), and
    # the original Python function, of which only a weak reference is kept. Returns nothing.
    self.weakrefself_target__ = target
    self.weakrefself_func__ = weakref.ref(original_python_function)

  @property
  def target(self):
    # [FUNCTION DOC]
    # Property returning the live bound object (instance or class) by dereferencing the stored
    # reference; yields nothing/null value if the referent has been garbage collected.
    return self.weakrefself_target__()

  @property
  def target_class(self):
    # [FUNCTION DOC]
    # Property returning the class associated with the target: the target itself when it is a
    # class (class-method case), otherwise the class of the instance. Raises an attribute error
    # if the referent has been collected and is therefore nothing.
    true_self = self.weakrefself_target__()
    if tf_inspect.isclass(true_self):
      # Class method
      return true_self
    else:
      return true_self.__class__

  def call(self, args, kwargs):
    # [FUNCTION DOC]
    # Invokes the original function with the live target as its first argument followed by the
    # given positional arguments (a sequence) and keyword arguments (a mapping). Returns the
    # function's result. Raises a type error if the function or target has been garbage collected,
    # since a missing callable would be invoked.
    wrapped_fn = self.weakrefself_func__()
    return wrapped_fn(self.weakrefself_target__(), *args, **kwargs)
