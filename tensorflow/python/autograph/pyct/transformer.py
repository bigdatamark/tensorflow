# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Base infrastructure for autograph source-code transformers. Provides: AnalysisLevel (which
# static analyses have been run), Context (per-conversion information), EntityInfo (record about
# the function or class being converted), private state-tracking helpers (_StateStack, _State),
# NodeStateTracker (shared helpers: scoped state, block visiting, debug printing), Base (an AST
# to AST transformer that tracks origin info and fixes expression wrappers) and CodeGenerator
# (an AST to string visitor that builds a source map). Converters and analyses subclass these.
# No import-time side effects.
"""A node transformer that includes utilities for SCT."""

import collections
import enum

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates


class AnalysisLevel(enum.IntEnum):
  # [CLASS DOC]
  # Integer enumeration ordering the static analyses available on a tree: none, activity,
  # definedness and liveness (0 to 3). Higher values imply the lower analyses were run. Pure
  # constants, no behavior.

  NONE = 0
  ACTIVITY = 1
  DEFINEDNESS = 2
  LIVENESS = 3


# TODO(znado): Use namedtuple.
class Context(object):
  # [CLASS DOC]
  # Mutable, non-thread-safe bag of information shared by a transformation pass. Attributes:
  # info (immutable entity description), namer (generator of unique names), current_origin
  # (origin info of the last successfully processed node, starts as nothing, maintained by the
  # transformers for error reporting) and user (opaque user-supplied object passed through).
  """Contains information about a source code transformation.

  This object is mutable, and is updated during conversion. Not thread safe.

  Attributes:
    info: EntityInfo, immutable.
    namer: naming.Namer.
    current_origin: origin_info.OriginInfo, holds the OriginInfo of the last
      AST node to be processed successfully. Useful for error handling.
    user: An user-supplied context object. The object is opaque to the
      infrastructure, but will pe passed through to all custom transformations.
  """

  def __init__(self, info, namer, user_context):
    # [FUNCTION DOC]
    # Stores the entity info, the name generator and the opaque user context, and initializes
    # the current origin to nothing. Returns nothing.
    self.info = info
    self.namer = namer
    self.current_origin = None
    self.user = user_context


# TODO(mdan): Move to a standalone file.
class EntityInfo(
    collections.namedtuple(
        'EntityInfo',
        ('name', 'source_code', 'source_file', 'future_features', 'namespace'))
):
  # [CLASS DOC]
  # Immutable named record describing a function or class being converted, with fields name,
  # source_code, source_file, future_features (tuple of future-feature names the entity was
  # compiled with) and namespace (mapping of symbols visible to the entity, excluding
  # parameters). Adds no behavior.
  """Contains information about a Python entity.

  Immutable.

  Examples of entities include functions and classes.

  Attributes:
    name: The name that identifies this entity.
    source_code: The entity's source code.
    source_file: The entity's source file.
    future_features: Tuple[Text], the future features that this entity was
      compiled with. See
      https://docs.python.org/2/reference/simple_stmts.html#future.
    namespace: Dict[str, ], containing symbols visible to the entity (excluding
      parameters).
  """
  pass


class _StateStack(object):
  # [CLASS DOC]
  # HELPER: implementation detail of the state tracking used by transformers (see _State).
  # A stack of instances of one given type, usable as a context manager (entering pushes a new
  # instance, leaving pops it). Attribute reads and writes on the stack object are forwarded to
  # the instance at the top of the stack. If the type has a marker attribute meaning "no root",
  # the stack starts empty; otherwise one instance is pushed at construction. Reading or writing
  # attributes on an empty stack fails with an index error.
  """Templated context manager.

  This class provides syntactic sugar for a stack of objects of known
  type. It allows accessing attributes of the object at the top of the stack
  directly against this object, which allows for very terse syntax.

  For example, this code:

    stack = _StateStack(Foo)
    stack.enter()
    stack.bar

  Is equivalent to:

    stack = []
    stack.append(Foo())
    foo = stack[-1]
    foo.bar

  See _State for more on how this is used.

  Attributes:
    type: Any, the type of objects that this stack holds
    level: int, the current stack depth
    stack: List[Any], the actual stack
    value: Any, the instance of the object at the top of the stack
  """

  def __init__(self, type_):
    # [FUNCTION DOC]
    # HELPER: part of the state stack implementation.
    # Records the element type and creates the empty internal list, bypassing the attribute
    # forwarding. Unless the type declares it has no root, immediately pushes one instance.
    # Because we override __setattr__, we need to attach these attributes using
    # the superclass' setattr.
    object.__setattr__(self, 'type', type_)
    object.__setattr__(self, '_stack', [])
    if not hasattr(type_, 'no_root'):
      self.enter()

  def __enter__(self):
    # [FUNCTION DOC]
    # Context-manager entry: pushes a new instance and returns the stack object itself.
    self.enter()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    # [FUNCTION DOC]
    # Context-manager exit: pops the top instance. Exception details are ignored, so exceptions
    # propagate. Returns nothing.
    self.exit()

  def enter(self):
    # [FUNCTION DOC]
    # Pushes a new default-constructed instance of the element type onto the stack.
    self._stack.append(self.type())

  def exit(self):
    # [FUNCTION DOC]
    # Pops the top instance. Fails with an index error if the stack is empty.
    self._stack.pop()

  @property
  def stack(self):
    # [FUNCTION DOC]
    # Property returning the underlying list (not a copy).
    return self._stack

  @property
  def level(self):
    # [FUNCTION DOC]
    # Property returning the current depth (number of instances on the stack).
    return len(self._stack)

  @property
  def value(self):
    # [FUNCTION DOC]
    # Property returning the instance at the top of the stack; index error if empty.
    return self._stack[-1]

  def __iter__(self):
    # [FUNCTION DOC]
    # Iterates over the instances from the bottom of the stack to the top.
    return iter(self._stack)

  def __getattr__(self, key):
    # [FUNCTION DOC]
    # Called for attributes not found normally; forwards the lookup to the top instance.
    return getattr(self._stack[-1], key)

  def __setattr__(self, key, value):
    # [FUNCTION DOC]
    # Forwards every attribute assignment to the instance at the top of the stack, so the stack
    # object itself never gains attributes this way.
    setattr(self._stack[-1], key, value)


class _State(object):
  # [CLASS DOC]
  # HELPER: container behind the state attribute of NodeStateTracker. Maps a type to a state
  # stack of that type, creating the stack on first access, like a default-valued mapping.
  # This allows independent scoped state per type during tree walks.
  """Syntactic sugar for accessing an instance of a StateStack context manager.

  This structure offers syntactic sugar over a dict of stacks of objects
  of known type. These structures are useful to keep state during AST walks.
  Multiple different scopes can be tracked in parallel. For example:

    s = _State()

    s[foo].enter()
    s[bar].enter()  # this will not affect s[foo]

  Element access has special semantics:
    * keys are a data type
    * element values are _StateStack(type=key) objects
    * missing elements are automatically added, similarly to defaultdict

  For example, the following block :

    _State s
    s[Foo]

  Is equivalent to:

    s = {}
    if Foo not in s:
      s[Foo] = Foo()
    s[Foo]

  See Base for how it's used.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the empty internal mapping from types to state stacks.
    self._value = {}

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Returns the state stack for the given type key, creating and storing a new one if absent.
    if key not in self._value:
      self._value[key] = _StateStack(key)
    return self._value[key]


class NodeStateTracker(object):
  # [CLASS DOC]
  # Mixin-style base for code transformers and generators. Holds the conversion Context (ctx),
  # an unused line/column pair, and a state container giving independent scoped state per type
  # (see _State). Offers debugging printers and a block-visiting helper. Meant to be combined
  # with a node visitor class (as Base and CodeGenerator do), since it relies on a visit method.
  # KNOWN ISSUE: the docstring mentions enter/exit_local_scope and set/get_local, which do not
  # exist here; scoped state is only available through the state attribute.
  """Base class for general-purpose Python code transformation.

  This abstract class provides helpful functions, like state tracking within
  the scope of arbitrary node, helpers for processing code blocks, debugging,
  mapping of transformed code to original code, and others.

  Scope-local state tracking: to keep state across nodes, at the level of
  (possibly nested) scopes, use enter/exit_local_scope and set/get_local.
  You must call enter/exit_local_scope manually, but the transformer detects
  when they are not properly paired.

  The transformer allows keeping state across calls that is local
  to arbitrary nodes and their descendants, using the self.state attribute.
  Multiple independent scopes are allowed and automatically constructed.

  For example, to keep track of the `If` node that encloses any `Name` node,
  one can write:

  ```
    class FooType(object):

      def __init__(self):
        self.foo_property = None

    class DummyTransformer(NodeStateTracker, ast.NodeTransformer):

      def visit_If(self, node):
        self.state[FooType].enter()
        self.state[FooType].foo_property = node
        node = self.veneric_visit(node)
        self.state[FooType].exit()
        return node

      def visit_Name(self, node):
        self.state[FooType].foo_property  # will hold the innermost enclosing if
  ```

  Alternatively, the `enter()`/`exit()` calls can be managed by a `with`
  statement:

  ```
      def visit_If(self, node):
        with self.state[FooType] as foo:
          foo.foo_property = node
          return self.generic_visit(node)
  ```
  """

  # TODO(mdan): Document all extra features.

  def __init__(self, ctx):
    # [FUNCTION DOC]
    # Stores the Context object (ctx) and creates the empty scoped-state container. Also sets
    # two unused position counters to zero. Subclasses must call this. Returns nothing.
    """Initialize the transformer.

    Subclasses should call this.

    Args:
      ctx: A Context object.
    """
    self._lineno = 0
    self._col_offset = 0
    self.ctx = ctx

    # Allows scoping of local variables to keep state across calls to visit_*
    # methods. Multiple scope hierarchies may exist and are keyed by tag. A
    # scope is valid at one or more nodes and all its children. Scopes created
    # in child nodes supersede their parent. Scopes are isolated from one
    # another.
    self.state = _State()

  def debug_print(self, node):
    # [FUNCTION DOC]
    # HELPER: debugging aid. Prints a formatted dump of the tree to standard output (only when
    # Python is not running with optimizations) and returns the node unchanged, so it can be
    # wrapped around a return value.
    """Helper method useful for debugging. Prints the AST."""
    if __debug__:
      print(pretty_printer.fmt(node))
    return node

  def debug_print_src(self, node):
    # [FUNCTION DOC]
    # HELPER: debugging aid. Prints the node converted back to source code to standard output
    # (only when not running with optimizations) and returns the node unchanged.
    """Helper method useful for debugging. Prints the AST as code."""
    if __debug__:
      print(parser.unparse(node))
    return node

  def visit_block(self, nodes, before_visit=None, after_visit=None):
    # [FUNCTION DOC]
    # Visits each statement of a block in order with the transformer's visit method and collects
    # the results into a new list. nodes is an iterable of nodes (nothing returns nothing).
    # before_visit (optional callback with no arguments) is called before each item.
    # after_visit (optional callback taking the replacement node and returning a pair of new
    # node and new destination list) is called after each non-empty replacement; a non-null
    # destination redirects all following results to that list. Replacements may be single
    # nodes or lists/tuples (flattened); empty results are dropped. Returns the results list,
    # excluding nodes relocated by the callback.
    """A more powerful version of generic_visit for statement blocks.

    An example of a block is the body of an if statement.

    This function allows specifying a postprocessing callback (the
    after_visit argument) argument which can be used to move nodes to a new
    destination. This is done by after_visit by returning a non-null
    second return value, e.g. return new_node, new_destination.

    For example, a transformer could perform the following move:

        foo()
        bar()
        baz()

        foo()
        if cond:
          bar()
          baz()

    The above could be done with a postprocessor of this kind:

        def after_visit(node):
          if node_is_function_call(bar):
            new_container_node = build_cond()
            new_container_node.body.append(node)
            return new_container_node, new_container_node.body
          else:
            # Once we set a new destination, all subsequent items will be
            # moved to it, so we don't need to explicitly handle baz.
            return node, None

    Args:
      nodes: enumerable of AST node objects. If None, the function returns None.
      before_visit: optional callable that is called before visiting each item
        in nodes
      after_visit: optional callable that takes in an AST node and returns a
        tuple (new_node, new_destination). It is called after visiting each item
        in nodes. Is used in the same was as the visit_* methods: new_node will
        replace the node; if not None, new_destination must be a list, and
        subsequent nodes will be placed in this list instead of the list
        returned by visit_block.

    Returns:
      A list of AST node objects containing the transformed items from nodes,
      except those nodes that have been relocated using after_visit.
    """
    if nodes is None:
      return None

    results = []
    node_destination = results
    for node in nodes:
      if before_visit:
        # TODO(mdan): We can modify node here too, if ever needed.
        before_visit()

      replacement = self.visit(node)

      if after_visit and replacement:
        replacement, new_destination = after_visit(replacement)
      else:
        new_destination = None

      if replacement:
        if isinstance(replacement, (list, tuple)):
          node_destination.extend(replacement)
        else:
          node_destination.append(replacement)

      # Allow the postprocessor to reroute the remaining nodes to a new list.
      if new_destination is not None:
        node_destination = new_destination
    return results


# TODO(mdan): Rename to PythonCodeTransformer.
class Base(NodeStateTracker, gast.NodeTransformer):
  # [CLASS DOC]
  # Base class for converters that rewrite a tree into another tree. Combines the state tracking
  # helpers with the node transformer, and overrides visit to: validate that the input is a
  # node, skip nodes flagged to skip processing, track the current origin in the context,
  # collapse expression wrappers around replaced assignments, and give replacement nodes the
  # origin info of the node they replaced. Instances are built with a Context.
  """Base class for general-purpose Python-to-Python code transformation.

  This is an extension of ast.NodeTransformer that provides the additional
  functions offered by NodeStateTracker.
  """

  def create_assignment(self, target, expression):
    # [FUNCTION DOC]
    # Builds an assignment of expression to target from a template. target and expression are
    # AST nodes, names or qualified names as accepted by the template replace function. Returns
    # the list of nodes produced by that function.
    template = """
      target = expression
    """
    return templates.replace(template, target=target, expression=expression)

  # TODO(mdan): Remove.
  def apply_to_single_assignments(self, targets, values, apply_fn):
    # [FUNCTION DOC]
    # Breaks a possibly unpacking assignment into individual (target, value) pairs and calls
    # apply_fn(target, value) for each; returns nothing. targets is a node or list/tuple of
    # nodes; values is the assigned node; apply_fn is a two-argument callback. Tuple/list
    # targets are recursed element-wise: if values is a tuple/list, the matching element is
    # used (an index error if shorter), otherwise a subscript of values by position is
    # synthesized. Other targets are passed through as they are.
    # KNOWN ISSUE: the synthesized subscript uses a store context and a plain integer index
    # rather than a constant node, which looks inconsistent for a value being read.
    """Applies a function to each individual assignment.

    This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
    It tries to break down the unpacking if possible. In effect, it has the same
    effect as passing the assigned values in SSA form to apply_fn.

    Examples:

    The following will result in apply_fn(a, c), apply_fn(b, d):

        a, b = c, d

    The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

        a, b = c

    The following will result in apply_fn(a, (b, c)):

        a = b, c

    It uses the visitor pattern to allow subclasses to process single
    assignments individually.

    Args:
      targets: list, tuple of or individual AST node. Should be used with the
        targets field of an ast.Assign node.
      values: an AST node.
      apply_fn: a function of a single argument, which will be called with the
        respective nodes of each single assignment. The signature is
        apply_fn(target, value), no return value.
    """
    if not isinstance(targets, (list, tuple)):
      targets = (targets,)
    for target in targets:
      if isinstance(target, (gast.Tuple, gast.List)):
        for i in range(len(target.elts)):
          target_el = target.elts[i]
          if isinstance(values, (gast.Tuple, gast.List)):
            value_el = values.elts[i]
          else:
            value_el = gast.Subscript(values, i, ctx=gast.Store())
          self.apply_to_single_assignments(target_el, value_el, apply_fn)
      else:
        # TODO(mdan): Look into allowing to rewrite the AST here.
        apply_fn(target, values)

  def visit(self, node):
    # [FUNCTION DOC]
    # Visits one node, wrapping the standard dispatch. Raises ValueError if node is not a tree
    # node (e.g. a list; use the block visitor). Returns the node unchanged if it is flagged to
    # skip processing. Otherwise sets the context's current origin from the node's origin
    # annotation (restored afterwards, even on error), dispatches, and then: if an expression
    # statement's value was replaced by a list, tuple or assignment, returns that replacement in
    # place of the wrapper; and any result differing from the input that lacks origin info
    # inherits the replaced node's origin (or the parent's). Returns the result, which may be a
    # node, a list, or nothing.
    if not isinstance(node, gast.AST):
      # This is not that uncommon a mistake: various node bodies are lists, for
      # example, posing a land mine for transformers that need to recursively
      # call `visit`.  The error needs to be raised before the exception handler
      # below is installed, because said handler will mess up if `node` is not,
      # in fact, a node.
      msg = ('invalid value for "node": expected "ast.AST", got "{}"; to'
             ' visit lists of nodes, use "visit_block" instead').format(
                 type(node))
      raise ValueError(msg)

    if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
      return node

    parent_origin = self.ctx.current_origin
    if anno.hasanno(node, anno.Basic.ORIGIN):
      self.ctx.current_origin = anno.getanno(node, anno.Basic.ORIGIN)

    try:
      processing_expr_node = isinstance(node, gast.Expr)
      if processing_expr_node:
        entry_expr_value = node.value

      result = super(Base, self).visit(node)

      # Adjust for consistency: replacing the value of an Expr with
      # an Assign node removes the need for the Expr node.
      if (processing_expr_node and isinstance(result, gast.Expr) and
          (result.value is not entry_expr_value)):
        # When the replacement is a list, it is assumed that the list came
        # from a template that contained a number of statements, which
        # themselves are standalone and don't require an enclosing Expr.
        if isinstance(result.value,
                      (list, tuple, gast.Assign, gast.AugAssign)):
          result = result.value

      # By default, all replacements receive the origin info of the replaced
      # node.
      if result is not node and result is not None:
        inherited_origin = anno.getanno(
            node, anno.Basic.ORIGIN, default=parent_origin)
        if inherited_origin is not None:
          nodes_to_adjust = result
          if isinstance(result, (list, tuple)):
            nodes_to_adjust = result
          else:
            nodes_to_adjust = (result,)
          for n in nodes_to_adjust:
            if not anno.hasanno(n, anno.Basic.ORIGIN):
              anno.setanno(n, anno.Basic.ORIGIN, inherited_origin)
    finally:
      self.ctx.current_origin = parent_origin

    return result


class CodeGenerator(NodeStateTracker, gast.NodeVisitor):
  # [CLASS DOC]
  # Base class for generators that turn a tree into text. Holds an output buffer (read via
  # code_buffer, appended via emit) and a source map from output ranges to origin info. Subclasses
  # supply per-node handlers that emit text. The visit override tracks origin info and records
  # the map entries.
  """Base class for general-purpose Python-to-string code transformation.

  Similar to Base, but outputs arbitrary strings instead of a Python AST.

  This uses the same visitor mechanism that the standard NodeVisitor uses,
  meaning that subclasses write handlers for the different kinds of nodes.
  New code is generated using the emit method, which appends to a code buffer
  that can be afterwards obtained from code_buffer.

  Example:

    class SimpleCodeGen(CodeGenerator):

      def visitIf(self, node):
        self.emit('if ')
        self.visit(node.test)
        self.emit(' { ')
        self.visit(node.body)
        self.emit(' } else { ')
        self.visit(node.orelse)
        self.emit(' } ')

    node = ast.parse(...)
    gen = SimpleCodeGen()
    gen.visit(node)
    # gen.code_buffer contains the resulting code
  """

  def __init__(self, ctx):
    # [FUNCTION DOC]
    # Initializes the base tracker with the Context, then creates an empty output text buffer and
    # an empty source map (mapping from output position pairs to origin info).
    super(CodeGenerator, self).__init__(ctx)

    self._output_code = ''
    self.source_map = {}

  def emit(self, code):
    # [FUNCTION DOC]
    # Appends the given text to the output buffer. Returns nothing.
    self._output_code += code

  @property
  def code_buffer(self):
    # [FUNCTION DOC]
    # Property returning all text emitted so far.
    return self._output_code

  def visit(self, node):
    # [FUNCTION DOC]
    # Visits one node and records origin info for the text it produced. Returns nothing at all
    # for nodes flagged to skip processing. Otherwise sets the context's current origin from the
    # node (restored afterwards, even on error), dispatches to the handler, and, if the buffer
    # length changed, stores the node's (or inherited) origin in the source map keyed by the
    # before and after buffer lengths. Returns the handler's result.
    # KNOWN ISSUE: the key is stored as (length before, length after) but the change test uses
    # the before minus the after difference, which is never positive-meaningful as a range check
    # (it is nonzero whenever text was emitted, so it works only incidentally).
    if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
      return

    parent_origin = self.ctx.current_origin
    eof_before = len(self._output_code)
    if anno.hasanno(node, anno.Basic.ORIGIN):
      self.ctx.current_origin = anno.getanno(node, anno.Basic.ORIGIN)

    try:
      ret = super(CodeGenerator, self).visit(node)

      # By default, all replacements receive the origin info of the replaced
      # node.
      eof_after = len(self._output_code)
      if eof_before - eof_after:
        inherited_origin = anno.getanno(
            node, anno.Basic.ORIGIN, default=parent_origin)
        if inherited_origin is not None:
          self.source_map[(eof_before, eof_after)] = inherited_origin
      return ret
    finally:
      self.ctx.current_origin = parent_origin
