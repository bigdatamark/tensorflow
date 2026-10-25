# [MODULE DOC]
# Decorators and accessors that control how the TensorFlow API doc generator treats an object.
# Each decorator works by setting a hidden marker attribute on the object (so it needs an object
# that accepts new attributes, and must sit below property, classmethod or staticmethod
# decorators) and returns the same object. Markers cover: deprecated, inheritable header text, do
# not document, do not document in this class or any subclass, only document in the defining class
# (for subclass implementers), force documenting a private member, and re-enable documentation in
# a class and its subclasses. Also defines aliases for some of these. The doc generator reads the
# markers; this module itself has no import-time side effects beyond defining constants.
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
"""Documentation control decorators."""

from typing import Optional, TypeVar

T = TypeVar("T")


_DEPRECATED = "_tf_docs_deprecated"


def set_deprecated(obj: T) -> T:
  # [FUNCTION DOC]
  # Decorator: marks an object as deprecated for the doc generator by setting a hidden marker
  # attribute. Input: obj, any object that accepts new attributes. Returns the same object.
  # Raises an attribute error if the object does not allow attributes to be set.
  """Explicitly tag an object as deprecated for the doc generator."""
  setattr(obj, _DEPRECATED, None)
  return obj


_INHERITABLE_HEADER = "_tf_docs_inheritable_header"


def inheritable_header(text):
  # [FUNCTION DOC]
  # Decorator factory: given header text, returns a class decorator that stores the text on the
  # class in a hidden attribute, which subclasses inherit, for the doc generator to show as a
  # header. Input: text, a string. Returns the decorator. Also exported under the alias header.

  def _wrapped(cls):
    # [FUNCTION DOC]
    # HELPER: the class decorator returned by the header decorator factory.
    # Sets the hidden header attribute (using the captured text) on the class and returns the
    # class. Raises an attribute error if the class does not allow attribute assignment.
    setattr(cls, _INHERITABLE_HEADER, text)
    return cls

  return _wrapped


def get_inheritable_header(obj) -> Optional[str]:
  # [FUNCTION DOC]
  # Reads back the header text set by the header decorator (including values inherited from base
  # classes). Input: obj, any object. Returns the string, or null if none was set. Also
  # exported under the alias get_header.
  return getattr(obj, _INHERITABLE_HEADER, None)


header = inheritable_header
get_header = get_inheritable_header

_DO_NOT_DOC = "_tf_docs_do_not_document"


def do_not_generate_docs(obj: T) -> T:
  # [FUNCTION DOC]
  # Decorator: marks an object (class, method or other attribute) so that the doc generator skips
  # it; the mark is not inherited, so subclasses overriding the method are still documented.
  # Input: obj, an object that accepts new attributes. Returns the same object.
  # Raises an attribute error if attributes cannot be set.
  """A decorator: Do not generate docs for this object.

  For example the following classes:

  ```
  class Parent(object):
    def method1(self):
      pass
    def method2(self):
      pass

  class Child(Parent):
    def method1(self):
      pass
    def method2(self):
      pass
  ```

  Produce the following api_docs:

  ```
  /Parent.md
    # method1
    # method2
  /Child.md
    # method1
    # method2
  ```

  This decorator allows you to skip classes or methods:

  ```
  @do_not_generate_docs
  class Parent(object):
    def method1(self):
      pass
    def method2(self):
      pass

  class Child(Parent):
    @do_not_generate_docs
    def method1(self):
      pass
    def method2(self):
      pass
  ```

  This will only produce the following docs:

  ```
  /Child.md
    # method2
  ```

  Note: This is implemented by adding a hidden attribute on the object, so it
  cannot be used on objects which do not allow new attributes to be added. So
  this decorator must go *below* `@property`, `@classmethod`,
  or `@staticmethod`:

  ```
  class Example(object):
    @property
    @do_not_generate_docs
    def x(self):
      return self._x
  ```

  Args:
    obj: The object to hide from the generated docs.

  Returns:
    obj
  """
  setattr(obj, _DO_NOT_DOC, None)
  return obj


_DO_NOT_DOC_INHERITABLE = "_tf_docs_do_not_doc_inheritable"


def do_not_doc_inheritable(obj: T) -> T:
  # [FUNCTION DOC]
  # Decorator: marks a class attribute so it is not documented in the defining class or in any
  # subclass, even those that override it (the generator searches the base class chain for the
  # mark). Input: obj, an object accepting new attributes. Returns the same object.
  # Raises an attribute error if attributes cannot be set.
  """A decorator: Do not generate docs for this method.

  This version of the decorator is "inherited" by subclasses. No docs will be
  generated for the decorated method in any subclass. Even if the sub-class
  overrides the method.

  For example, to ensure that `method1` is **never documented** use this
  decorator on the base-class:

  ```
  class Parent(object):
    @do_not_doc_inheritable
    def method1(self):
      pass
    def method2(self):
      pass

  class Child(Parent):
    def method1(self):
      pass
    def method2(self):
      pass
  ```
  This will produce the following docs:

  ```
  /Parent.md
    # method2
  /Child.md
    # method2
  ```

  When generating docs for a class's arributes, the `__mro__` is searched and
  the attribute will be skipped if this decorator is detected on the attribute
  on any class in the `__mro__`.

  Note: This is implemented by adding a hidden attribute on the object, so it
  cannot be used on objects which do not allow new attributes to be added. So
  this decorator must go *below* `@property`, `@classmethod`,
  or `@staticmethod`:

  ```
  class Example(object):
    @property
    @do_not_doc_inheritable
    def x(self):
      return self._x
  ```

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """
  setattr(obj, _DO_NOT_DOC_INHERITABLE, None)
  return obj


_FOR_SUBCLASS_IMPLEMENTERS = "_tf_docs_tools_for_subclass_implementers"


def for_subclass_implementers(obj: T) -> T:
  # [FUNCTION DOC]
  # Decorator: marks an attribute to be documented only in the class that defines it (grouped with
  # abstract methods), and skipped in subclasses. Intended for public methods that subclass
  # authors implement but callers should not invoke directly. Input: obj, an object accepting new
  # attributes. Returns the same object. Raises an attribute error if attributes cannot be set.
  """A decorator: Only generate docs for this method in the defining class.

  Also group this method's docs with and `@abstractmethod` in the class's docs.

  No docs will generated for this class attribute in sub-classes.

  The canonical use case for this is `tf.keras.layers.Layer.call`: It's a
  public method, essential for anyone implementing a subclass, but it should
  never be called directly.

  Works on method, or other class-attributes.

  When generating docs for a class's arributes, the `__mro__` is searched and
  the attribute will be skipped if this decorator is detected on the attribute
  on any **parent** class in the `__mro__`.

  For example:

  ```
  class Parent(object):
    @for_subclass_implementers
    def method1(self):
      pass
    def method2(self):
      pass

  class Child1(Parent):
    def method1(self):
      pass
    def method2(self):
      pass

  class Child2(Parent):
    def method1(self):
      pass
    def method2(self):
      pass
  ```

  This will produce the following docs:

  ```
  /Parent.md
    # method1
    # method2
  /Child1.md
    # method2
  /Child2.md
    # method2
  ```

  Note: This is implemented by adding a hidden attribute on the object, so it
  cannot be used on objects which do not allow new attributes to be added. So
  this decorator must go *below* `@property`, `@classmethod`,
  or `@staticmethod`:

  ```
  class Example(object):
    @property
    @for_subclass_implementers
    def x(self):
      return self._x
  ```

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """
  setattr(obj, _FOR_SUBCLASS_IMPLEMENTERS, None)
  return obj


do_not_doc_in_subclasses = for_subclass_implementers

_DOC_PRIVATE = "_tf_docs_doc_private"


def doc_private(obj: T) -> T:
  # [FUNCTION DOC]
  # Decorator: forces documentation of a private (underscore-prefixed) method or function, which
  # the generator normally omits. Input: obj, an object accepting new attributes. Returns the
  # same object. Raises an attribute error if attributes cannot be set.
  # KNOWN ISSUE: the docstring describes the argument as the class attribute to hide from the
  # generated docs, which is the opposite of what the decorator does (it asks to include it).
  """A decorator: Generates docs for private methods/functions.

  For example:

  ```
  class Try:

    @doc_controls.doc_private
    def _private(self):
      ...
  ```

  As a rule of thumb, private(beginning with `_`) methods/functions are
  not documented.

  This decorator allows to force document a private method/function.

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """

  setattr(obj, _DOC_PRIVATE, None)
  return obj


_DOC_IN_CURRENT_AND_SUBCLASSES = "_tf_docs_doc_in_current_and_subclasses"


def doc_in_current_and_subclasses(obj: T) -> T:
  # [FUNCTION DOC]
  # Decorator: overrides the do-not-document-in-subclasses mark of a parent for the decorated
  # attribute, so it is documented in this class and in every class inheriting from it.
  # Input: obj, an object accepting new attributes. Returns the same object.
  # Raises an attribute error if attributes cannot be set.
  # KNOWN ISSUE: the docstring describes the argument as the class attribute to hide from the
  # generated docs, which is wrong; the decorator causes it to be shown.
  """Overrides `do_not_doc_in_subclasses` decorator.

  If this decorator is set on a child class's method whose parent's method
  contains `do_not_doc_in_subclasses`, then that will be overriden and the
  child method will get documented. All classes inherting from the child will
  also document that method.

  For example:

  ```
  class Parent:
    @do_not_doc_in_subclasses
    def method1(self):
      pass
    def method2(self):
      pass

  class Child1(Parent):
    @doc_in_current_and_subclasses
    def method1(self):
      pass
    def method2(self):
      pass

  class Child2(Parent):
    def method1(self):
      pass
    def method2(self):
      pass

  class Child11(Child1):
    pass
  ```

  This will produce the following docs:

  ```
  /Parent.md
    # method1
    # method2
  /Child1.md
    # method1
    # method2
  /Child2.md
    # method2
  /Child11.md
    # method1
    # method2
  ```

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """

  setattr(obj, _DOC_IN_CURRENT_AND_SUBCLASSES, None)
  return obj
