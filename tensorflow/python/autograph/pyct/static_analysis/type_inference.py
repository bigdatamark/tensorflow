# [MODULE DOC]
# Static type inference for autograph. Defines an abstract Resolver interface (supplied by
# callers) that looks up types of names, literals, arguments, calls, slices, comparisons and
# operators; a statement-level inferrer that annotates expression nodes with sets of possible
# types and static values; a control-flow-graph analyzer that propagates type maps across
# statements, including closure effects; a per-function visitor; and the resolve entry point.
# Requires prior reaching function definitions analysis. No import-time side effects other than
# defining a sentinel for missing values.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Type inference.

This analysis annotates all symbols nodes of an AST with type information
extracted from static sources:
 * type annotations
 * global and local symbols visible to the function at analysis time
 * literals

Important: This analysis is static, and does not detect dynamic type changes.
The analysis attempts to use the values of external symbols, if available. These
values are also considered static for the purpose of analysis.

Requires reaching function definitions analysis.
"""

import itertools

from typing import Any, Callable, Dict, Set

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos


class Resolver(object):
  # [CLASS DOC]
  # Abstract interface through which the inference asks the embedding application for type
  # information. Every method raises a not-implemented error and must be overridden by
  # subclasses. Methods receive a namespace (names to actual values) and a types namespace
  # (names to inferred types) and return sets of types or nothing when unknown.
  """Resolver objects handle the process of looking up actual names and types.

  Unless noted otherwise, all resolve_* methods:
    * have a first namespace argument, mapping string to actual values
    * have a second types_namespace argument, mapping string to actual inferred
      types
    * specify names as QN objects
    * specify types as a Set of inferred types

  Unless noted otherwise, all resolve_* methods must return either:
    * a set of `type` objects
    * None
  """

  def res_name(self, ns, types_ns, name):
    # [FUNCTION DOC]
    # Abstract hook: resolve the type and static value of an external (closure or global)
    # variable given the namespaces and qualified name. Should return a pair of type set and
    # static value, or nothing for unknown. Base version raises a not-implemented error.
    """Resolves the type/value an external (e.g. closure, global) variable.

    Args:
      ns: namespace
      types_ns: types namespace
      name: symbol name

    Returns:
      Tuple (type, static_value). The first element is the type to use for
      inference. The second is the static value to use. Return None to treat it
      as unknown.
    """
    raise NotImplementedError('subclasses must implement')

  def res_value(self, ns, value):
    # [FUNCTION DOC]
    # Abstract hook: return the set of types of a literal or static value in the namespace.
    # Base version raises a not-implemented error.
    """Resolves the type a literal or static value."""
    raise NotImplementedError('subclasses must implement')

  def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
    # [FUNCTION DOC]
    # Abstract hook: return the set of types of a function argument, given the function name,
    # argument name, its annotation (if any) and whether the function is local. Base version
    # raises a not-implemented error.
    """Resolves the type of a (possibly annotated) function argument.

    Args:
      ns: namespace
      types_ns: types namespace
      f_name: str, the function name
      name: str, the argument name
      type_anno: the type annotating the argument, if any
      f_is_local: bool, whether the function is a local function
    Returns:
      Set of the argument types.
    """
    raise NotImplementedError('subclasses must implement')

  def res_call(self, ns, types_ns, node, f_type, args, keywords):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring calls node a string function name, but the inferrer passes the
    # whole call syntax node.
    # Abstract hook: resolve the return type of a call to an external function. Receives the
    # call node, the callee's types if known, and lists of argument and keyword value types.
    # Should return a pair of return type set and a mapping from names to type sets modelling
    # side effects. Base version raises a not-implemented error.
    """Resolves the return type an external function or method call.

    Args:
      ns: namespace
      types_ns: types namespace
      node: str, the function name
      f_type: types of the actual function being called, if known
      args: types of each respective argument in node.args
      keywords: types of each respective argument in node.keywords

    Returns:
      Tuple (return_type, side_effect_types). The first element is just the
      return types of the function. The second element is a map from
      argument names to sets of types, and allow modelling side effects of
      functions (for example via global or nonlocal).
    """
    raise NotImplementedError('subclasses must implement')

  # TODO(mdan): Clean this up.
  def res_slice(self, ns, types_ns, node_or_slice, value, slice_):
    # [FUNCTION DOC]
    # Abstract hook: resolve the type of indexing a value of the given types with the given slice
    # types. Also used for tuple unpacking, where an integer index replaces the node. Base
    # version raises a not-implemented error.
    """Resolves the return type of slice operation."""
    raise NotImplementedError('subclasses must implement')

  def res_compare(self, ns, types_ns, node, left, right):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says unary operation, but this hook is for comparisons.
    # Abstract hook: resolve the result type of a comparison given the left types and a list of
    # types for each comparator. Base version raises a not-implemented error.
    """Resolves the return type of a unary operation."""
    raise NotImplementedError('subclasses must implement')

  def res_unop(self, ns, types_ns, node, opnd):
    # [FUNCTION DOC]
    # Abstract hook: resolve the result type of a unary operation given the operand types. Base
    # version raises a not-implemented error.
    """Resolves the return type of a unary operation."""
    raise NotImplementedError('subclasses must implement')

  def res_binop(self, ns, types_ns, node, left, right):
    # [FUNCTION DOC]
    # Abstract hook: resolve the result type of a binary operation given left and right operand
    # types. Base version raises a not-implemented error.
    """Resolves the return type of a binary operation."""
    raise NotImplementedError('subclasses must implement')

  def res_list_literal(self, ns, elt_types):
    # [FUNCTION DOC]
    # Abstract hook: resolve the type of a list literal given a tuple of element type sets. Base
    # version raises a not-implemented error.
    """Resolves the type of a list literal from its elements."""
    raise NotImplementedError('subclasses must implement')


class _TypeMap(object):
  # [CLASS DOC]
  # HELPER: dataflow lattice value used by the type inference Analyzer.
  # Value type holding a mapping from symbol names to the set of types they may have at a program
  # point. Supports copy construction, equality and union (merging type sets per symbol).
  """Abstraction for the state of the CFG walk for type inference.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    types: Dict[qual_names.QN, Set[Type]], mapping symbols to the set of
      possible types.
  """

  def __init__(self, init_from=None):
    # [FUNCTION DOC]
    # Constructor. init_from is an optional other type map (asserted) to copy, including copies
    # of its type sets. Nothing gives an empty map.
    if init_from:
      assert isinstance(init_from, _TypeMap)
      self.types = {
          s: set(other_types) for s, other_types in init_from.types.items()
      }
    else:
      self.types = {}

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Returns true if both maps have the same symbols and equal type sets for each.
    if frozenset(self.types.keys()) != frozenset(other.types.keys()):
      return False
    ret = all(self.types[s] == other.types[s] for s in self.types)
    return ret

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Returns the negation of equality.
    return not self.__eq__(other)

  def __or__(self, other):
    # [FUNCTION DOC]
    # Union of two type maps (other must be a type map, asserted). Returns a new map where each
    # symbol's type set is the union of both; operands are unchanged.
    assert isinstance(other, _TypeMap)
    result = _TypeMap(self)
    for s, other_types in other.types.items():
      if s not in result.types:
        self_types = set()
        result.types[s] = self_types
      else:
        self_types = result.types[s]
      self_types.update(other_types)
    return result

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debug string showing the symbol-to-types mapping.
    return 'SymbolTable {}'.format(self.types)


NO_VALUE = object()


class StmtInferrer(gast.NodeVisitor):
  # [CLASS DOC]
  # Node visitor that infers types for the expressions of a single statement. Each visit method
  # returns a set of possible types or nothing if unknown, and the visit override stores the
  # result on the node as a tuple. State: resolver, scope of the function, namespace of actual
  # values, closure types, the incoming type map, new_symbols (types of symbols bound by this
  # statement, read by the analyzer) and rtype (the right-hand side type while visiting
  # assignment targets). Only some node kinds are supported; others fall back to generic
  # traversal and give no type.
  """Runs type inference on a single AST statement.

  This visitor annotates most nodes with type information. It also sets types
  for the symbols modified by this statement in its types_out property.

  Note: this inferrer is able to capture side effects of functions, however,
  these side effects will not be applied to the current expression. Doing so
  would create too much of a dependence on the runtime's internal rules about
  execution order.
  Example:

    def f():
      nonlocal a
      a = 1
      return a

    a = 0.0
    b = f() + a  # a = float; side effect of f() ignored
    print(a)  # a = int; side effect of f() accounted for
  """

  def __init__(self,
               resolver: Resolver,
               scope: activity.Scope,
               namespace: Dict[qual_names.QN, Any],
               closure_types: Dict[qual_names.QN, Set[Any]],
               types_in: _TypeMap):
    # [FUNCTION DOC]
    # Constructor. resolver is the type resolver; scope is the activity scope of the enclosing
    # function; namespace maps names to actual values; closure_types maps names to type sets
    # coming from enclosing functions; types_in is the type map at statement entry. Initializes
    # an empty new_symbols mapping and no right-hand-side type.
    self.resolver = resolver
    self.scope = scope
    self.namespace = namespace
    self.closure_types = closure_types
    self.types_in = types_in
    self.new_symbols = {}

    # rvalue type. This property is set when encountering an assign operation,
    # so that visiting nodes with Store ctx (typically found on left side of
    # assignments) can infer the type they should receive.
    self.rtype = None

  def visit(self, node):
    # [FUNCTION DOC]
    # Dispatches to the specific visit method, validates (in debug mode) that the result is a set
    # or nothing, and if a result exists stores it on the node as a tuple of types. Returns the
    # set of types or nothing. Raises a value error if a method returns a non-set.
    types = super().visit(node)
    if __debug__:
      self._check_set(types)
    if types is not None:
      # TODO(mdan): Normalize by removing subtypes.
      anno.setanno(node, anno.Static.TYPES, tuple(types))
    return types

  def _check_set(self, value):
    # [FUNCTION DOC]
    # HELPER: debug validation of resolver and visitor results.
    # Raises a value error (naming the resolver) if the value is neither nothing nor a set.
    if value is not None and not isinstance(value, set):
      raise ValueError('{} method expected to return set, got {}'.format(
          self.resolver, value))

  def visit_Constant(self, node):
    # [FUNCTION DOC]
    # Returns the resolver's type set for the literal value of the constant node.
    types = self.resolver.res_value(self.namespace, node.value)
    if __debug__:
      self._check_set(types)
    return types

  def _apply_unpacking(self, node):
    # [FUNCTION DOC]
    # HELPER: shared by tuple and list target visits.
    # For a tuple or list in store context, if a right-hand-side type is set, visits each element
    # with the type obtained by indexing that type with the element position (via the resolver),
    # then restores the original right-hand-side type and returns it. With no right-hand-side
    # type returns nothing. Asserts the node is in store context.
    assert isinstance(node.ctx, gast.Store)
    if self.rtype is not None:
      original_stype = self.rtype
      # TODO(mdan): Find a better way to express unpacking.
      i_type = self.resolver.res_value(self.namespace, 0)
      for i, elt in enumerate(node.elts):
        self.rtype = self.resolver.res_slice(
            self.namespace, self.types_in.types, i, original_stype, i_type)
        self.visit(elt)
      self.rtype = original_stype
      return original_stype
    return None

  def visit_Tuple(self, node):
    # [FUNCTION DOC]
    # In load context, visits the elements; if any is unknown returns nothing, else returns the
    # set of all combinations (as tuples) of the element types. In store context performs
    # unpacking of the assigned type.
    if isinstance(node.ctx, gast.Load):
      elt_types = ()
      for elt in node.elts:
        types_ = self.visit(elt)
        if types_ is None:
          return None
        elt_types += (types_,)
      return set(itertools.product(*elt_types))
    return self._apply_unpacking(node)

  def visit_List(self, node):
    # [FUNCTION DOC]
    # In load context, visits the elements and returns the resolver's list literal type for the
    # tuple of element types (unknown elements are passed through as nothing). In store context
    # performs unpacking.
    if isinstance(node.ctx, gast.Load):
      elt_types = tuple(self.visit(elt) for elt in node.elts)
      return self.resolver.res_list_literal(self.namespace, elt_types)
    return self._apply_unpacking(node)

  def visit_Set(self, node):
    # [FUNCTION DOC]
    # KNOWN ISSUE: set literals are unsupported; always raises a not-implemented error.
    raise NotImplementedError()

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Infers the type set of a name by context. Load: uses the incoming type map; if unknown and
    # the name is not bound locally (or is nonlocal), uses closure types, else asks the resolver
    # for the name, annotating the node with any static value returned. Param: asks the resolver
    # for the argument type using the annotation, function name and whether the function is
    # local, and records it as a new symbol. Store: records the right-hand-side type as a new
    # symbol and returns it. Any other context fails an assertion.
    name = anno.getanno(node, anno.Basic.QN)

    if isinstance(node.ctx, gast.Load):
      types = self.types_in.types.get(name, None)
      if types is None:
        if (name not in self.scope.bound) or (name in self.scope.nonlocals):
          # TODO(mdan): Test with global variables.
          if name in self.closure_types:
            types = self.closure_types[name]
          else:
            types, value = self.resolver.res_name(
                self.namespace, self.types_in.types, name)
            if value is not None:
              anno.setanno(node, anno.Static.VALUE, value)

    elif isinstance(node.ctx, gast.Param):
      # The direct parent it the whole function scope. See activity.py.
      f_is_local = self.scope.parent.parent is not None

      type_name = anno.getanno(node.annotation, anno.Basic.QN, None)
      types = self.resolver.res_arg(self.namespace, self.types_in.types,
                                    self.scope.function_name, name, type_name,
                                    f_is_local)
      if types is not None:
        self.new_symbols[name] = types

    elif isinstance(node.ctx, gast.Store):
      if self.rtype is not None:
        self.new_symbols[name] = self.rtype
      types = self.rtype

    else:
      assert False, 'unknown ctx'

    if __debug__:
      self._check_set(types)

    return types

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when the static attribute lookup fails it passes the whole incoming type map
    # object to the resolver's name hook, whereas other calls pass its inner mapping.
    # Infers the type of an attribute access. Visits the base expression. If the base has a known
    # static value, reads the attribute from it (falling back to resolving the full qualified
    # name through the resolver when missing). Otherwise tries the attribute on each possible
    # base type: unknown, empty, missing or unstable (not identical across types) gives nothing.
    # On success annotates the node with the static value and returns the resolver's type set
    # for it.
    parent_types = self.visit(node.value)

    # Attempt to use the static value if known.
    parent_value = anno.Static.VALUE.of(node.value, None)
    if parent_value is not None:
      static_value = getattr(parent_value, node.attr, NO_VALUE)

      if static_value is NO_VALUE:
        # Unexpected failure to resolve attribute. Ask the resolver about the
        # full name instead.
        types, static_value = self.resolver.res_name(
            self.namespace, self.types_in, anno.Basic.QN.of(node))
        anno.setanno(node, anno.Static.VALUE, static_value)
        if __debug__:
          self._check_set(types)
        return types

    else:
      # Fall back to the type if that is known.
      if parent_types is None:
        return None

      inferred_values = [getattr(t, node.attr, None) for t in parent_types]
      if not inferred_values:
        return None

      static_value = inferred_values[0]
      if static_value is None:
        return None

      if any(v is not static_value for v in inferred_values[1:]):
        # Static value not stable, assume it's dynamic.
        return None

    types = self.resolver.res_value(self.namespace, static_value)
    anno.setanno(node, anno.Static.VALUE, static_value)

    if __debug__:
      self._check_set(types)

    return types

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Records the function's name as a new symbol whose type is a callable returning the declared
    # return types (resolved through the resolver) or the any type if no return annotation or
    # unresolved. Raises a not-implemented error if the function has decorators. Returns nothing.
    f_name = qual_names.QN(node.name)

    if node.decorator_list:
      raise NotImplementedError('decorators: {}'.format(node.decorator_list))

    ret_types = None
    if node.returns:
      ret_types, _ = self.resolver.res_name(
          self.namespace, self.types_in.types, anno.Basic.QN.of(node.returns))
      if __debug__:
        self._check_set(ret_types)

    if ret_types is None:
      ret_types = {Any}

    f_types = set()
    for rt in ret_types:
      f_types.add(Callable[[Any], rt])

    self.new_symbols[f_name] = f_types
    # The definition of a function is an expression, hence has no return value.
    return None

  def _resolve_typed_callable(self, f_types, arg_types, keyword_types):
    # [FUNCTION DOC]
    # HELPER: return-type lookup for locally defined functions.
    # For each callable type in f_types, takes the last type argument as the return type (any if
    # no type arguments); relies on undocumented typing internals. Raises a not-implemented
    # error for a non-callable type. Argument types are ignored. Returns a pair of the return
    # type set and nothing for side effects.
    ret_types = set()
    for t in f_types:

      if isinstance(t, Callable):
        # Note: these are undocumented - may be version-specific!
        # Callable[[x], y]: __args__ are (x, y)
        args = t.__args__
        if args:
          ret_types.add(args[-1])
        else:
          ret_types.add(Any)
      else:
        raise NotImplementedError('callable type {}'.format(type(t)))

    # Side effects can not be inferred based on type alone.
    side_effects = None
    return ret_types, side_effects

  def visit_Call(self, node):
    # [FUNCTION DOC]
    # Infers the return type of a call. Visits the callee, positional arguments and keyword
    # values. If the callee is bound in the current scope, uses its locally recorded callable
    # type (no type info gives nothing); otherwise asks the resolver's call hook with the
    # callee's types and argument types. In debug mode validates that the result is a set and
    # side effects are a mapping from qualified names to sets, raising a value error otherwise.
    # Side effects are merged into new_symbols. Returns the return type set or nothing.
    self.visit(node.func)

    f_name = anno.Basic.QN.of(node.func)
    arg_types = [self.visit(a) for a in node.args]
    keyword_types = [self.visit(kw.value) for kw in node.keywords]

    if f_name in self.scope.bound:
      # Local function, use local type definitions, if available.
      f_type = self.types_in.types.get(f_name, None)
      if f_type is None:
        # No static type info available, nothing more to do.
        ret_type, side_effects = None, None
      else:
        ret_type, side_effects = self._resolve_typed_callable(
            f_type, arg_types, keyword_types)

    else:
      # Nonlocal function, resolve externally.
      f_type = anno.Static.TYPES.of(node.func, None)
      ret_type, side_effects = self.resolver.res_call(self.namespace,
                                                      self.types_in.types, node,
                                                      f_type, arg_types,
                                                      keyword_types)

    if __debug__:
      self._check_set(ret_type)
      if side_effects:
        if not isinstance(side_effects, dict):
          raise ValueError(
              'side effects must be dict, got {}'.format(side_effects))
        for k, v in side_effects.items():
          if not isinstance(k, qual_names.QN):
            raise ValueError('side effect keys must be QNs, got {}'.format(k))
          self._check_set(v)

    if side_effects:
      self.new_symbols.update(side_effects)
    return ret_type

  def visit_Expr(self, node):
    # [FUNCTION DOC]
    # Visits the expression statement's value and returns its type set (or nothing).
    return self.visit(node.value)

  def visit_Assign(self, node):
    # [FUNCTION DOC]
    # Infers the right-hand-side type, stores it as the current right-hand-side type, visits each
    # target so store-context names (and unpacked elements) receive it, then clears it. Returns
    # nothing.
    self.rtype = self.visit(node.value)

    for t in node.targets:
      self.visit(t)

    self.rtype = None

  def visit_Subscript(self, node):
    # [FUNCTION DOC]
    # Visits the value and index; if either is unknown returns nothing; otherwise returns the
    # resolver's slice type for the node and both type sets.
    val_types = self.visit(node.value)
    slice_types = self.visit(node.slice)

    if val_types is None or slice_types is None:
      return None

    types = self.resolver.res_slice(
        self.namespace, self.types_in.types, node, val_types, slice_types)

    if __debug__:
      self._check_set(types)

    return types

  def visit_Compare(self, node):
    # [FUNCTION DOC]
    # Visits the left operand and all comparators; if any is unknown returns nothing; otherwise
    # returns the resolver's comparison type given the left types and list of comparator types.
    left_types = self.visit(node.left)
    right_types = [self.visit(c) for c in node.comparators]

    if left_types is None or any(t is None for t in right_types):
      return None

    types = self.resolver.res_compare(
        self.namespace, self.types_in.types, node, left_types, right_types)

    if __debug__:
      self._check_set(types)

    return types

  def visit_BinOp(self, node):
    # [FUNCTION DOC]
    # Visits both operands; if either is unknown returns nothing; otherwise returns the
    # resolver's binary operation type for the node and operand types.
    left_types = self.visit(node.left)
    right_types = self.visit(node.right)

    if left_types is None or right_types is None:
      return None

    types = self.resolver.res_binop(
        self.namespace, self.types_in.types, node, left_types, right_types)

    if __debug__:
      self._check_set(types)

    return types

  def visit_UnaryOp(self, node):
    # [FUNCTION DOC]
    # Visits the operand; if unknown returns nothing; otherwise returns the resolver's unary
    # operation type for the node and operand types.
    opnd_types = self.visit(node.operand)

    if opnd_types is None:
      return None

    types = self.resolver.res_unop(
        self.namespace, self.types_in.types, node, opnd_types)

    if __debug__:
      self._check_set(types)

    return types


class Analyzer(cfg.GraphVisitor):
  # [CLASS DOC]
  # Forward dataflow visitor over one function's control flow graph that propagates type maps
  # between statements. For each node, runs a statement inferrer on its syntax node and merges
  # newly bound symbols into the output map. Also records types visible to nested functions on
  # their definition nodes (closure types). State: resolver, namespace, scope, closure types and
  # context types (closure types for names not bound locally, injected at the graph entry).
  """CFG visitor that propagates type information across statements."""

  def __init__(self, graph, resolver, namespace, scope, closure_types):
    # [FUNCTION DOC]
    # Constructor. graph is the function's control flow graph; resolver the type resolver;
    # namespace maps names to values; scope is the function's activity scope; closure_types maps
    # names to type sets from enclosing functions. Computes context types as the closure types
    # of names not bound in the scope (nothing if empty).
    """Creates a new analyzer.

    Args:
      graph: cfg.Graph
      resolver: Resolver
      namespace: Dict[str, Any]
      scope: activity.Scope
      closure_types: Dict[QN, Set]
    """
    super(Analyzer, self).__init__(graph)
    self.resolver = resolver
    self.namespace = namespace
    self.scope = scope
    self.closure_types = closure_types

    context_types = {
        n: t for n, t in closure_types.items() if n not in scope.bound
    }
    if context_types:
      self.context_types = _TypeMap()
      self.context_types.types = context_types
    else:
      self.context_types = None

  def init_state(self, _):
    # [FUNCTION DOC]
    # Initial dataflow value for any node (argument ignored): a fresh empty type map.
    return _TypeMap()

  def _update_closure_types(self, ast_node, types):
    # [FUNCTION DOC]
    # HELPER: communicates types to nested functions.
    # Merges the given type map into the closure types annotation of the function node, creating
    # the annotation if missing; type sets are unioned per symbol. Mutates the annotation.
    existing_types = anno.Static.CLOSURE_TYPES.of(ast_node, None)

    if existing_types is None:
      existing_types = {}
      anno.Static.CLOSURE_TYPES.add_to(ast_node, existing_types)

    for k, v in types.types.items():
      if k in existing_types:
        existing_types[k].update(v)
      else:
        existing_types[k] = set(v)

  def visit_node(self, node):
    # [FUNCTION DOC]
    # Transfer function for one CFG node. Input is the union of predecessors' outputs (plus
    # context types at the graph entry). Runs a statement inferrer on the node's syntax node and
    # updates the output with the symbols it bound. Then, if the node has scope info, for each
    # reaching function definition whose name is read by the node, merges the output types into
    # that function node's closure types. Stores in/out maps and returns true if the output
    # changed.
    prev_types_out = self.out[node]

    types_in = _TypeMap()
    for n in node.prev:
      types_in |= self.out[n]
    if (self.context_types is not None) and (node is self.graph.entry):
      types_in |= self.context_types

    types_out = _TypeMap(types_in)
    ast_node = node.ast_node

    inferrer = StmtInferrer(self.resolver, self.scope, self.namespace,
                            self.closure_types, types_in)
    inferrer.visit(ast_node)
    types_out.types.update(inferrer.new_symbols)

    reaching_fndefs = anno.Static.DEFINED_FNS_IN.of(ast_node)
    node_scope = anno.Static.SCOPE.of(ast_node, None)
    if node_scope is not None:
      # TODO(mdan): Check that it's actually safe to skip nodes without scope.
      reads = {str(qn) for qn in node_scope.read}
      for def_node in reaching_fndefs:
        if def_node.name in reads:
          self._update_closure_types(def_node, types_out)

    self.in_[node] = types_in
    self.out[node] = types_out

    return prev_types_out != types_out


class FunctionVisitor(transformer.Base):
  # [CLASS DOC]
  # AST transformer that runs the type inference Analyzer separately on each function found,
  # outer functions first, using the function's graph, scope annotation and closure types
  # annotation. State: graphs (function nodes to graphs) and the resolver.
  """AST visitor that applies type inference to each function separately."""

  def __init__(self, source_info, graphs, resolver):
    # [FUNCTION DOC]
    # Constructor. source_info describes the source (including the namespace of values); graphs
    # maps function nodes to control flow graphs; resolver is the type resolver.
    super(FunctionVisitor, self).__init__(source_info)
    self.graphs = graphs
    self.resolver = resolver

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Runs a forward type analysis over the function's graph using its arguments-and-body scope
    # and any closure types previously attached (default empty), then recursively processes
    # nested functions in the body. Returns the node, annotated.
    subgraph = self.graphs[node]
    scope = anno.getanno(node, annos.NodeAnno.ARGS_AND_BODY_SCOPE)
    closure_types = anno.getanno(node, anno.Static.CLOSURE_TYPES, {})

    analyzer = Analyzer(subgraph, self.resolver, self.ctx.info.namespace, scope,
                        closure_types)
    analyzer.visit_forward()

    # Recursively process any remaining subfunctions.
    node.body = self.visit_block(node.body)

    return node


def resolve(node, source_info, graphs, resolver):
  # [FUNCTION DOC]
  # Entry point for type inference. node is the AST to analyze, source_info describes the
  # source and namespace, graphs maps function nodes to control flow graphs, resolver is a
  # Resolver implementation. Annotates nodes with types, static values and closure types in
  # place and returns the annotated tree.
  """Performs type inference.

  Args:
    node: ast.AST
    source_info: transformer.SourceInfo
    graphs: Dict[ast.FunctionDef, cfg.Graph]
    resolver: Resolver

  Returns:
    ast.AST
  """
  visitor = FunctionVisitor(source_info, graphs, resolver)
  node = visitor.visit(node)
  return node
