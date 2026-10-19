# [MODULE DOC]
# HELPER: Internal glue that lets Keras Model, Layer and TensorBoard classes pick their v1
# (graph-mode) or v2 (eager) implementation at instantiation time.
# Defines lazily loaded references to the v1 and v2 training, base layer and callbacks modules
# (so importing this module does not import the engine), three selector base classes that swap
# the concrete class when an instance is created, a function deciding whether v2 behavior applies,
# a class-swapping function, a guard raising an error for legacy graph usage, and a v1 type check.
# Import-time side effect: only creation of the lazy module placeholders.
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
# pylint: disable=protected-access
"""Utilities for Keras classes with v1 and v2 versions."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader

# TODO(b/134426265): Switch back to single-quotes once the issue
# with copybara is fixed.
# pylint: disable=g-inconsistent-quotes
training = LazyLoader(
    "training", globals(),
    "tensorflow.python.keras.engine.training")
training_v1 = LazyLoader(
    "training_v1", globals(),
    "tensorflow.python.keras.engine.training_v1")
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
base_layer_v1 = LazyLoader(
    "base_layer_v1", globals(),
    "tensorflow.python.keras.engine.base_layer_v1")
callbacks = LazyLoader(
    "callbacks", globals(),
    "tensorflow.python.keras.callbacks")
callbacks_v1 = LazyLoader(
    "callbacks_v1", globals(),
    "tensorflow.python.keras.callbacks_v1")


# pylint: enable=g-inconsistent-quotes


class ModelVersionSelector(object):
  # [CLASS DOC]
  # HELPER: Mixin base for the Keras Model classes. It has no state; its instance-creation hook
  # replaces the class being instantiated (and its ancestors) by the v1 or v2 Model class
  # depending on the current execution mode.
  """Chooses between Keras v1 and v2 Model class."""

  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Instance-creation hook. Ignores the constructor arguments. Decides whether v2 applies,
    # swaps the class hierarchy to the matching Model class (this modifies the base classes of
    # the class being instantiated, a global side effect), and returns a new bare instance of
    # the resulting class; the normal initializer then runs.
    use_v2 = should_use_v2()
    cls = swap_class(cls, training.Model, training_v1.Model, use_v2)  # pylint: disable=self-cls-assignment
    return super(ModelVersionSelector, cls).__new__(cls)


class LayerVersionSelector(object):
  # [CLASS DOC]
  # HELPER: Mixin base for Keras Layer classes. Stateless; its instance-creation hook swaps the
  # class hierarchy to the v1 or v2 base Layer class depending on the execution mode.
  """Chooses between Keras v1 and v2 Layer class."""

  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Instance-creation hook. Ignores constructor arguments. Decides whether v2 applies, swaps the
    # class hierarchy to the matching Layer class (mutating the base classes of the class being
    # created) and returns a new bare instance of the resulting class.
    use_v2 = should_use_v2()
    cls = swap_class(cls, base_layer.Layer, base_layer_v1.Layer, use_v2)  # pylint: disable=self-cls-assignment
    return super(LayerVersionSelector, cls).__new__(cls)


class TensorBoardVersionSelector(object):
  # [CLASS DOC]
  # HELPER: Mixin base for the TensorBoard callback classes. Stateless; its instance-creation hook
  # swaps the class hierarchy to the v1 or v2 TensorBoard callback depending on execution mode.
  """Chooses between Keras v1 and v2 TensorBoard callback class."""

  def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Instance-creation hook for the TensorBoard callback. Decides whether v2 applies and swaps
    # the class hierarchy accordingly (mutating base classes). Special case: when the started
    # class is the v1 callback itself but the swap yields the v2 callback (which is not a
    # subclass of v1, so the initializer would not run), it directly constructs and returns a v2
    # instance with the given arguments. Otherwise returns a new bare instance of the chosen class.
    use_v2 = should_use_v2()
    start_cls = cls
    cls = swap_class(start_cls, callbacks.TensorBoard, callbacks_v1.TensorBoard,
                     use_v2)
    if start_cls == callbacks_v1.TensorBoard and cls == callbacks.TensorBoard:
      # Since the v2 class is not a subclass of the v1 class, __init__ has to
      # be called manually.
      return cls(*args, **kwargs)
    return super(TensorBoardVersionSelector, cls).__new__(cls)


def should_use_v2():
  # [FUNCTION DOC]
  # Returns true if v2 (eager-style) classes should be used: always when executing eagerly; when
  # eager execution is enabled outside functions, true unless the current default graph is a
  # wrapped-function graph (code inside such a wrapper is treated as v1); false otherwise.
  # Reads global execution state only; no inputs and no side effects.
  """Determine if v1 or v2 version should be used."""
  if context.executing_eagerly():
    return True
  elif ops.executing_eagerly_outside_functions():
    # Check for a v1 `wrap_function` FuncGraph.
    # Code inside a `wrap_function` is treated like v1 code.
    graph = ops.get_default_graph()
    if (getattr(graph, "name", False) and
        graph.name.startswith("wrapped_function")):
      return False
    return True
  else:
    return False


def swap_class(cls, v2_cls, v1_cls, use_v2):
  # [FUNCTION DOC]
  # HELPER: Core of the version selector classes.
  # Inputs: cls the class to adjust, v2_cls and v1_cls the two alternative base classes, and
  # use_v2 a boolean choosing which to prefer. If cls is the plain root object class it is
  # returned unchanged; if cls is one of the two alternatives, the chosen one is returned.
  # Otherwise each direct base class that derives from the unwanted alternative is recursively
  # swapped, and the base class list of cls is overwritten in place (side effect, permanent for
  # that class). Returns cls.
  """Swaps in v2_cls or v1_cls depending on graph mode."""
  if cls == object:
    return cls
  if cls in (v2_cls, v1_cls):
    return v2_cls if use_v2 else v1_cls

  # Recursively search superclasses to swap in the right Keras class.
  new_bases = []
  for base in cls.__bases__:
    if ((use_v2 and issubclass(base, v1_cls)
         # `v1_cls` often extends `v2_cls`, so it may still call `swap_class`
         # even if it doesn't need to. That being said, it may be the safest
         # not to over optimize this logic for the sake of correctness,
         # especially if we swap v1 & v2 classes that don't extend each other,
         # or when the inheritance order is different.
         or (not use_v2 and issubclass(base, v2_cls)))):
      new_base = swap_class(base, v2_cls, v1_cls, use_v2)
    else:
      new_base = base
    new_bases.append(new_base)
  cls.__bases__ = tuple(new_bases)
  return cls


def disallow_legacy_graph(cls_name, method_name):
  # [FUNCTION DOC]
  # Guard for methods that cannot run in graph mode on objects built under eager mode. Inputs:
  # the class name and the method name, used only to build the message. Raises a value error
  # when eager execution is not enabled outside functions; otherwise returns nothing.
  if not ops.executing_eagerly_outside_functions():
    error_msg = (
        "Calling `{cls_name}.{method_name}` in graph mode is not supported "
        "when the `{cls_name}` instance was constructed with eager mode "
        "enabled. Please construct your `{cls_name}` instance in graph mode or"
        " call `{cls_name}.{method_name}` with eager mode enabled.")
    error_msg = error_msg.format(cls_name=cls_name, method_name=method_name)
    raise ValueError(error_msg)


def is_v1_layer_or_model(obj):
  # [FUNCTION DOC]
  # Returns true if obj is an instance of the v1 base Layer or the v1 Model class. Triggers lazy
  # loading of the v1 modules on first use.
  return isinstance(obj, (base_layer_v1.Layer, training_v1.Model))
