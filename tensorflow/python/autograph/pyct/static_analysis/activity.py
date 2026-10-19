# [MODULE DOC]
# Activity analysis for AutoGraph. Defines Scope (tracks which symbols are read, modified, deleted,
# bound, declared global/nonlocal, used as annotations, or are function parameters within a block)
# and ActivityAnalyzer (a transformer that walks a syntax tree and attaches Scope objects to nodes
# using annotation keys). Requires that qualified-name annotations were already resolved on the
# tree. Exposes the resolve function as the entry point. No import-time side effects.
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
"""Activity analysis.

Requires qualified name annotations (see qual_names.py).
"""

import copy
import weakref

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno


class Scope(object):
  # [CLASS DOC]
  # Mutable-then-frozen record of symbol activity (read, modified, deleted, bound, globals,
  # nonlocals, annotations, params) for one code block or true Python scope. Scopes form a parent
  # chain. Built from the innermost block outward and frozen with finalize, which propagates
  # information to the parent. Holds only weak references to parameter owner nodes.
  """Encloses local symbol definition and usage information.

  This can track for instance whether a symbol is modified in the current scope.
  Note that scopes do not necessarily align with Python's scopes. For example,
  the body of an if statement may be considered a separate scope.

  Caution - the AST references held by this object are weak.

  Scope objects are mutable during construction only, and must be frozen using
  `Scope.finalize()` before use. Furthermore, a scope is consistent only after
  all its children have been frozen. While analysing code blocks, scopes are
  being gradually built, from the innermost scope outward. Freezing indicates
  that the analysis of a code block is complete. Once frozen, mutation is no
  longer allowed. `is_final` tracks whether the scope is frozen or not. Certain
  properties, like `referenced`, are only accurate when called on frozen scopes.

  Attributes:
    parent: Optional[Scope], the parent scope, if any.
    isolated: bool, whether the scope is a true Python scope (e.g. the scope of
      a function), or just a surrogate tracking an ordinary code block. Using
      the terminology of the Python 3 reference documentation, True roughly
      represents an actual scope, whereas False represents an ordinary code
      block.
    function_name: Optional[str], name of the function owning this scope.
    isolated_names: Set[qual_names.QN], identifiers that are isolated to this
      scope (even if the scope is not isolated).
    annotations: Set[qual_names.QN], identifiers used as type annotations
      in this scope.
    read: Set[qual_names.QN], identifiers read in this scope.
    modified: Set[qual_names.QN], identifiers modified in this scope.
    deleted: Set[qual_names.QN], identifiers deleted in this scope.
    bound: Set[qual_names.QN], names that are bound to this scope. See
      https://docs.python.org/3/reference/executionmodel.html#binding-of-names
      for a precise definition.
    globals: Set[qual_names.QN], names that are explicitly marked as global in
      this scope. Note that this doesn't include free read-only vars bound to
      global symbols.
    nonlocals: Set[qual_names.QN], names that are explicitly marked as nonlocal
      in this scope. Note that this doesn't include free read-only vars bound to
      global symbols.
    free_vars: Set[qual_names.QN], the free variables in this scope. See
      https://docs.python.org/3/reference/executionmodel.html for a precise
      definition.
    params: WeakValueDictionary[qual_names.QN, ast.Node], function arguments
      visible in this scope, mapped to the function node that defines them.
    enclosing_scope: Scope, the innermost isolated scope that is a transitive
      parent of this scope. May be the scope itself.
    referenced: Set[qual_names.QN], the totality of the symbols used by this
      scope and its parents.
    is_final: bool, whether the scope is frozen or not.

  Note - simple statements may never delete and modify a symbol at the same
  time. However, compound ones like if statements can. In that latter case, it's
  undefined whether the symbol is actually modified or deleted upon statement
  exit. Certain analyses like reaching definitions need to be careful about
  this.
  """

  # Note: this mutable-immutable pattern is used because using a builder would
  # have taken a lot more boilerplate.

  def __init__(self, parent, isolated=True, function_name=None):
    # [FUNCTION DOC]
    # Creates an empty, unfrozen scope. Inputs: parent scope or nothing; isolated flag (default
    # true) saying whether it is a real Python scope; optional owning function name. Initializes
    # all symbol sets empty and a weak-valued mapping of params. Returns nothing.
    """Create a new scope.

    Args:
      parent: A Scope or None.
      isolated: Whether the scope is isolated, that is, whether variables
        modified in this scope should be considered modified in the parent
        scope.
      function_name: Name of the function owning this scope.
    """
    self.parent = parent
    self.isolated = isolated
    self.function_name = function_name

    self.isolated_names = set()

    self.read = set()
    self.modified = set()
    self.deleted = set()

    self.bound = set()
    self.globals = set()
    self.nonlocals = set()
    self.annotations = set()

    self.params = weakref.WeakValueDictionary()

    # Certain fields can only be accessed after the scope and all its parent
    # scopes have been fully built. This field guards that.
    self.is_final = False

  @property
  def enclosing_scope(self):
    # [FUNCTION DOC]
    # Property: the innermost isolated scope among this scope and its parents (the parent if this
    # scope is non-isolated and has one, otherwise itself). Asserts the scope is frozen first.
    assert self.is_final
    if self.parent is not None and not self.isolated:
      return self.parent
    return self

  @property
  def referenced(self):
    # [FUNCTION DOC]
    # Property: set of symbols read by this scope united with those referenced by all parents.
    # Returns a new set. Only accurate on frozen scopes.
    if self.parent is not None:
      return self.read | self.parent.referenced
    return self.read

  @property
  def free_vars(self):
    # [FUNCTION DOC]
    # Property: symbols read but not bound in the enclosing isolated scope. Returns a new set.
    # Requires the scope to be frozen (via enclosing scope).
    enclosing_scope = self.enclosing_scope
    return enclosing_scope.read - enclosing_scope.bound

  def copy_from(self, other):
    # [FUNCTION DOC]
    # KNOWN ISSUE: globals and nonlocals are not copied, although other fields are.
    # Overwrites this unfrozen scope's isolated names, modified, read, deleted, bound, annotations
    # and params with shallow copies from another scope, recursing into the parent chain (the
    # other scope must have a parent when this one does). Asserts not frozen. Returns nothing.
    """Recursively copies the contents of this scope from another scope."""
    assert not self.is_final
    if self.parent is not None:
      assert other.parent is not None
      self.parent.copy_from(other.parent)
    self.isolated_names = copy.copy(other.isolated_names)
    self.modified = copy.copy(other.modified)
    self.read = copy.copy(other.read)
    self.deleted = copy.copy(other.deleted)
    self.bound = copy.copy(other.bound)
    self.annotations = copy.copy(other.annotations)
    self.params = copy.copy(other.params)

  @classmethod
  def copy_of(cls, other):
    # [FUNCTION DOC]
    # Class method: builds a new unfrozen scope (default isolated, no function name) whose content
    # and parent chain are copies of another scope. Returns the new scope.
    if other.parent is not None:
      assert other.parent is not None
      parent = cls.copy_of(other.parent)
    else:
      parent = None
    new_copy = cls(parent)
    new_copy.copy_from(other)
    return new_copy

  def merge_from(self, other):
    # [FUNCTION DOC]
    # KNOWN ISSUE: globals and nonlocals are not merged.
    # Unions the activity sets (isolated names, read, modified, bound, deleted, annotations,
    # params) of another scope into this unfrozen scope, recursing through parents. Returns nothing.
    """Adds all activity from another scope to this scope."""
    assert not self.is_final
    if self.parent is not None:
      assert other.parent is not None
      self.parent.merge_from(other.parent)
    self.isolated_names.update(other.isolated_names)
    self.read.update(other.read)
    self.modified.update(other.modified)
    self.bound.update(other.bound)
    self.deleted.update(other.deleted)
    self.annotations.update(other.annotations)
    self.params.update(other.params)

  def finalize(self):
    # [FUNCTION DOC]
    # Freezes the scope and pushes its activity to the parent. For non-isolated scopes, reads,
    # modifications, bindings (minus isolated names), globals, nonlocals and annotations go to the
    # parent. For isolated scopes only reads and annotations not bound locally go to the parent.
    # Asserts the scope and its parent are not frozen. Returns nothing.
    """Freezes this scope."""
    assert not self.is_final
    # TODO(mdan): freeze read, modified, bound.
    if self.parent is not None:
      assert not self.parent.is_final
      if not self.isolated:
        self.parent.read.update(self.read - self.isolated_names)
        self.parent.modified.update(self.modified - self.isolated_names)
        self.parent.bound.update(self.bound - self.isolated_names)
        self.parent.globals.update(self.globals)
        self.parent.nonlocals.update(self.nonlocals)
        self.parent.annotations.update(self.annotations)
      else:
        # TODO(mdan): This is not accurate.
        self.parent.read.update(self.read - self.bound)
        self.parent.annotations.update(self.annotations - self.bound)
    self.is_final = True

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a short text showing the read and modified symbols of the scope.
    return 'Scope{r=%s, w=%s}' % (tuple(self.read), tuple(self.modified))

  def mark_param(self, name, owner):
    # [FUNCTION DOC]
    # Records that the symbol name is a parameter of the owner function node (weakly held).
    # Mutates the params mapping. Returns nothing.
    # Assumption: all AST nodes have the same life span. This lets us use
    # a weak reference to mark the connection between a symbol node and the
    # function node whose argument that symbol is.
    self.params[name] = owner


class _Comprehension(object):
  # [CLASS DOC]
  # HELPER: per-level state object for ActivityAnalyzer while visiting comprehensions.
  # Holds the set of comprehension target symbols and a list-comprehension flag. The class-level
  # no-root marker tells the transformer state machinery it needs no root level.

  no_root = True

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: initializes comprehension state: list-comprehension flag false, empty target set.
    # TODO(mdan): Consider using an enum.
    self.is_list_comp = False
    self.targets = set()


class _FunctionOrClass(object):
  # [CLASS DOC]
  # HELPER: per-level state object for ActivityAnalyzer holding the function, lambda or class
  # definition node currently being visited.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: initializes the state with no node.
    self.node = None


class ActivityAnalyzer(transformer.Base):
  # [CLASS DOC]
  # Transformer that annotates syntax tree nodes with Scope objects describing symbol activity.
  # Holds the current scope, flags for augmented assignment, annotation visiting and
  # annotation-only tracking. Created with a transformer context and optional parent scope; run by
  # visiting a tree (see resolve). Uses transformer state stacks for comprehensions and
  # function/class nesting. Mutates the tree by setting annotations.
  """Annotates nodes with local scope information.

  See Scope.

  The use of this class requires that qual_names.resolve() has been called on
  the node. This class will ignore nodes have not been
  annotated with their qualified names.
  """

  def __init__(self, context, parent_scope=None):
    # [FUNCTION DOC]
    # Initializes the analyzer with the transformer context and an optional parent scope. Disables
    # skipping of nodes, creates the root isolated scope and resets the internal flags.
    super(ActivityAnalyzer, self).__init__(context)
    self.allow_skips = False
    self.scope = Scope(parent_scope, isolated=True)

    # Note: all these flags crucially rely on the respective nodes are
    # leaves in the AST, that is, they cannot contain other statements.
    self._in_aug_assign = False
    self._in_annotation = False
    self._track_annotations_only = False

  @property
  def _in_constructor(self):
    # [FUNCTION DOC]
    # HELPER: property telling whether the innermost visited definition is a method named
    # __init__ directly inside a class. Returns a boolean; false when nesting is too shallow
    # (the level check requires more than two levels).
    context = self.state[_FunctionOrClass]
    if context.level > 2:
      innermost = context.stack[-1].node
      parent = context.stack[-2].node
      return (isinstance(parent, gast.ClassDef) and
              (isinstance(innermost, gast.FunctionDef) and
               innermost.name == '__init__'))
    return False

  def _node_sets_self_attribute(self, node):
    # [FUNCTION DOC]
    # HELPER: returns true if the node has a qualified name that is an attribute of a symbol
    # named self. Assumes the first argument is called self (noted TODO in the code).
    if anno.hasanno(node, anno.Basic.QN):
      qn = anno.getanno(node, anno.Basic.QN)
      # TODO(mdan): The 'self' argument is not guaranteed to be called 'self'.
      if qn.has_attr and qn.parent.qn == ('self',):
        return True
    return False

  def _track_symbol(self, node, composite_writes_alter_parent=False):
    # [FUNCTION DOC]
    # HELPER: records the activity of a name/attribute/subscript node in the current scope based
    # on its context. Store: modified and bound (and read if inside augmented assignment; parent
    # also modified for composite names when the flag is set). Load: read (and annotation if in
    # annotation). Param: bound and marked as parameter of the current function. Del: read, bound,
    # deleted. Ignores nodes without qualified name, targets of enclosing comprehensions (a store
    # in a comprehension just adds a target), and everything outside annotations when only
    # annotations are tracked. Raises a value error for an unknown context. Returns nothing.
    if self._track_annotations_only and not self._in_annotation:
      return

    # A QN may be missing when we have an attribute (or subscript) on a function
    # call. Example: a().b
    if not anno.hasanno(node, anno.Basic.QN):
      return
    qn = anno.getanno(node, anno.Basic.QN)

    # When inside a comprehension, ignore reads to any of the comprehensions's
    # targets. This includes attributes or slices of those arguments.
    for l in self.state[_Comprehension]:
      if qn in l.targets:
        return
      if qn.owner_set & set(l.targets):
        return

    if isinstance(node.ctx, gast.Store):
      # In comprehensions, modified symbols are the comprehension targets.
      if self.state[_Comprehension].level > 0:
        self.state[_Comprehension].targets.add(qn)
        return

      self.scope.modified.add(qn)
      self.scope.bound.add(qn)
      if qn.is_composite and composite_writes_alter_parent:
        self.scope.modified.add(qn.parent)
      if self._in_aug_assign:
        self.scope.read.add(qn)

    elif isinstance(node.ctx, gast.Load):
      self.scope.read.add(qn)
      if self._in_annotation:
        self.scope.annotations.add(qn)

    elif isinstance(node.ctx, gast.Param):
      self.scope.bound.add(qn)
      self.scope.mark_param(qn, self.state[_FunctionOrClass].node)

    elif isinstance(node.ctx, gast.Del):
      # The read matches the Python semantics - attempting to delete an
      # undefined symbol is illegal.
      self.scope.read.add(qn)
      # Targets of del are considered bound:
      # https://docs.python.org/3/reference/executionmodel.html#binding-of-names
      self.scope.bound.add(qn)
      self.scope.deleted.add(qn)

    else:
      raise ValueError('Unknown context {} for node "{}".'.format(
          type(node.ctx), qn))

  def _enter_scope(self, isolated, f_name=None):
    # [FUNCTION DOC]
    # HELPER: pushes a new child scope (isolated flag, optional function name) as current scope.
    self.scope = Scope(self.scope, isolated=isolated, function_name=f_name)

  def _exit_scope(self):
    # [FUNCTION DOC]
    # HELPER: freezes the current scope, makes its parent current and returns the frozen scope.
    exited_scope = self.scope
    exited_scope.finalize()
    self.scope = exited_scope.parent
    return exited_scope

  def _exit_and_record_scope(self, node, tag=anno.Static.SCOPE):
    # [FUNCTION DOC]
    # HELPER: exits the current scope and stores it on the node under the given annotation key
    # (default: the generic scope key). Returns the exited scope.
    node_scope = self._exit_scope()
    anno.setanno(node, tag, node_scope)
    return node_scope

  def _process_statement(self, node):
    # [FUNCTION DOC]
    # HELPER: visits a simple statement inside a new non-isolated scope and records that scope on
    # the node. Returns the visited node.
    self._enter_scope(False)
    node = self.generic_visit(node)
    self._exit_and_record_scope(node)
    return node

  def _process_annotation(self, node):
    # [FUNCTION DOC]
    # HELPER: visits a type-annotation expression with the in-annotation flag set, then clears it.
    # Returns the visited node.
    self._in_annotation = True
    node = self.visit(node)
    self._in_annotation = False
    return node

  def visit_Import(self, node):
    # [FUNCTION DOC]
    # Visitor for import statements: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_ImportFrom(self, node):
    # [FUNCTION DOC]
    # Visitor for from-import statements: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_Global(self, node):
    # [FUNCTION DOC]
    # Visitor for global declarations: in a fresh scope, each named symbol is added to read and
    # globals. Records the scope on the node and returns the node.
    self._enter_scope(False)
    for name in node.names:
      qn = qual_names.QN(name)
      self.scope.read.add(qn)
      self.scope.globals.add(qn)
    self._exit_and_record_scope(node)
    return node

  def visit_Nonlocal(self, node):
    # [FUNCTION DOC]
    # Visitor for nonlocal declarations: in a fresh scope, each named symbol is added to read,
    # bound and nonlocals. Records the scope on the node and returns the node.
    self._enter_scope(False)
    for name in node.names:
      qn = qual_names.QN(name)
      self.scope.read.add(qn)
      self.scope.bound.add(qn)
      self.scope.nonlocals.add(qn)
    self._exit_and_record_scope(node)
    return node

  def visit_Expr(self, node):
    # [FUNCTION DOC]
    # Visitor for expression statements: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_Raise(self, node):
    # [FUNCTION DOC]
    # Visitor for raise statements: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_Return(self, node):
    # [FUNCTION DOC]
    # Visitor for return statements: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_Assign(self, node):
    # [FUNCTION DOC]
    # Visitor for assignments: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_AnnAssign(self, node):
    # [FUNCTION DOC]
    # Visitor for annotated assignments: in a fresh scope visits the target, the value (if
    # present; pure declarations have none and are not definitions) and the annotation (as an
    # annotation). Records the scope on the node and returns it.
    self._enter_scope(False)
    node.target = self.visit(node.target)
    if node.value is not None:
      # Can be None for pure declarations, e.g. `n: int`. This is a new thing
      # enabled by type annotations, but does not influence static analysis
      # (declarations are not definitions).
      node.value = self.visit(node.value)
    if node.annotation:
      node.annotation = self._process_annotation(node.annotation)
    self._exit_and_record_scope(node)
    return node

  def visit_AugAssign(self, node):
    # [FUNCTION DOC]
    # Visitor for augmented assignments: in a fresh scope visits the target with the
    # augmented-assignment flag set so the target is both read and modified, then the operator and
    # value. Records the scope on the node and returns it.
    # Special rules for AugAssign. Here, the AST only shows the target as
    # written, when it is in fact also read.
    self._enter_scope(False)

    self._in_aug_assign = True
    node.target = self.visit(node.target)
    self._in_aug_assign = False

    node.op = self.visit(node.op)
    node.value = self.visit(node.value)
    self._exit_and_record_scope(node)
    return node

  def visit_Delete(self, node):
    # [FUNCTION DOC]
    # Visitor for delete statements: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Visitor for names: processes an attached annotation (if any) as a type annotation, then
    # tracks the symbol according to its context. Returns the node.
    if node.annotation:
      node.annotation = self._process_annotation(node.annotation)
    self._track_symbol(node)
    return node

  def visit_alias(self, node):
    # [FUNCTION DOC]
    # Visitor for import aliases: marks the bound name (the alias, or the root of a dotted module
    # name when there is no alias) as modified and bound in the current scope. Returns the node.
    node = self.generic_visit(node)

    if node.asname is None:
      # Only the root name is a real symbol operation.
      qn = qual_names.QN(node.name.split('.')[0])
    else:
      qn = qual_names.QN(node.asname)

    self.scope.modified.add(qn)
    self.scope.bound.add(qn)
    return node

  def visit_Attribute(self, node):
    # [FUNCTION DOC]
    # Visitor for attribute access: visits children, then tracks the symbol. Inside a constructor,
    # writes to attributes of self also mark the parent (self) as modified. Returns the node.
    node = self.generic_visit(node)
    if self._in_constructor and self._node_sets_self_attribute(node):
      self._track_symbol(node, composite_writes_alter_parent=True)
    else:
      self._track_symbol(node)
    return node

  def visit_Subscript(self, node):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment says subscript writes also modify the parent, but the call does not
    # enable the parent-altering option, so only the subscript symbol itself is tracked.
    # Visitor for subscripts: visits children then tracks the symbol. Returns the node.
    node = self.generic_visit(node)
    # Subscript writes (e.g. a[b] = "value") are considered to modify
    # both the element itself (a[b]) and its parent (a).
    self._track_symbol(node)
    return node

  def visit_Print(self, node):
    # [FUNCTION DOC]
    # Visitor for print statements: visits the printed values in a fresh scope and records it
    # both as the generic scope and as the arguments scope. Returns the node.
    self._enter_scope(False)
    node.values = self.visit_block(node.values)
    node_scope = self._exit_and_record_scope(node)
    anno.setanno(node, NodeAnno.ARGS_SCOPE, node_scope)
    return node

  def visit_Assert(self, node):
    # [FUNCTION DOC]
    # Visitor for assert statements: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_Call(self, node):
    # [FUNCTION DOC]
    # Visitor for calls: visits positional and keyword arguments in a fresh scope recorded under
    # the arguments-scope key (star-args handling is a TODO), then visits the called function
    # expression in the enclosing scope. Returns the node.
    self._enter_scope(False)
    node.args = self.visit_block(node.args)
    node.keywords = self.visit_block(node.keywords)
    # TODO(mdan): Account starargs, kwargs
    self._exit_and_record_scope(node, tag=NodeAnno.ARGS_SCOPE)

    node.func = self.visit(node.func)
    return node

  def _process_block_node(self, node, block, scope_name):
    # [FUNCTION DOC]
    # HELPER: visits a list of statements in a fresh non-isolated scope and records that scope on
    # the node under the given key. Returns the node.
    self._enter_scope(False)
    block = self.visit_block(block)
    self._exit_and_record_scope(node, tag=scope_name)
    return node

  def _process_parallel_blocks(self, parent, children):
    # [FUNCTION DOC]
    # HELPER: processes alternative blocks (such as body and else-branch) of a compound statement.
    # Checkpoints the current scope, then for each (block, scope key) pair restores the checkpoint,
    # processes the block and snapshots the result; finally merges all snapshots into the current
    # scope. Returns the parent node.
    # Because the scopes are not isolated, processing any child block
    # modifies the parent state causing the other child blocks to be
    # processed incorrectly. So we need to checkpoint the parent scope so that
    # each child sees the same context.
    before_parent = Scope.copy_of(self.scope)
    after_children = []
    for child, scope_name in children:
      self.scope.copy_from(before_parent)
      parent = self._process_block_node(parent, child, scope_name)
      after_child = Scope.copy_of(self.scope)
      after_children.append(after_child)
    for after_child in after_children:
      self.scope.merge_from(after_child)
    return parent

  def _process_comprehension(self,
                             node,
                             is_list_comp=False,
                             is_dict_comp=False):
    # [FUNCTION DOC]
    # HELPER: visits a comprehension within a comprehension state level. Generators are visited
    # first so their targets are known, then the key and value (dictionary comprehension) or the
    # element. Flags: list comprehension and dictionary comprehension (both default false).
    # Returns the node.
    with self.state[_Comprehension] as comprehension_:
      comprehension_.is_list_comp = is_list_comp
      # Note: it's important to visit the generators first to properly account
      # for the variables local to these generators. Example: `x` is local to
      # the expression `z for x in y for z in x`.
      node.generators = self.visit_block(node.generators)
      if is_dict_comp:
        node.key = self.visit(node.key)
        node.value = self.visit(node.value)
      else:
        node.elt = self.visit(node.elt)
      return node

  def visit_comprehension(self, node):
    # [FUNCTION DOC]
    # Visitor for one generator clause: visits the iterable, then the target (so reads of the
    # target are ignored), then the rest. Returns the node.
    # It is important to visit children in this order so that the reads to
    # the target name are appropriately ignored.
    node.iter = self.visit(node.iter)
    node.target = self.visit(node.target)
    return self.generic_visit(node)

  def visit_DictComp(self, node):
    # [FUNCTION DOC]
    # Visitor for dictionary comprehensions; delegates to comprehension processing.
    return self._process_comprehension(node, is_dict_comp=True)

  def visit_ListComp(self, node):
    # [FUNCTION DOC]
    # Visitor for list comprehensions; delegates to comprehension processing with list flag set.
    return self._process_comprehension(node, is_list_comp=True)

  def visit_SetComp(self, node):
    # [FUNCTION DOC]
    # Visitor for set comprehensions; delegates to comprehension processing.
    return self._process_comprehension(node)

  def visit_GeneratorExp(self, node):
    # [FUNCTION DOC]
    # Visitor for generator expressions; delegates to comprehension processing.
    return self._process_comprehension(node)

  def visit_ClassDef(self, node):
    # [FUNCTION DOC]
    # Visitor for class definitions. One non-isolated scope records the class name as modified and
    # bound plus decorators, bases and keywords, and is attached to the node. A separate isolated
    # scope covers the class body (generic visit) and is not recorded. Returns the node.
    with self.state[_FunctionOrClass] as fn:
      fn.node = node
      # The ClassDef node itself has a Scope object that tracks the creation
      # of its name, along with the usage of any decorator accompanying it.
      self._enter_scope(False)
      node.decorator_list = self.visit_block(node.decorator_list)
      self.scope.modified.add(qual_names.QN(node.name))
      self.scope.bound.add(qual_names.QN(node.name))
      node.bases = self.visit_block(node.bases)
      node.keywords = self.visit_block(node.keywords)
      self._exit_and_record_scope(node)

      # A separate Scope tracks the actual class definition.
      self._enter_scope(True)
      node = self.generic_visit(node)
      self._exit_scope()
      return node

  def _visit_node_list(self, nodes):
    # [FUNCTION DOC]
    # HELPER: visits each node in a list, keeping empty entries as nothing. Returns the new list.
    return [(None if n is None else self.visit(n)) for n in nodes]

  def _visit_arg_annotations(self, node):
    # [FUNCTION DOC]
    # HELPER: visits default values of arguments normally, then visits the argument declarations
    # with annotation-only tracking so only annotation reads are recorded. Returns the node.
    node.args.kw_defaults = self._visit_node_list(node.args.kw_defaults)
    node.args.defaults = self._visit_node_list(node.args.defaults)
    self._track_annotations_only = True
    node = self._visit_arg_declarations(node)
    self._track_annotations_only = False
    return node

  def _visit_arg_declarations(self, node):
    # [FUNCTION DOC]
    # HELPER: visits positional-only, regular, variadic, keyword-only and keyword-variadic
    # argument nodes of a function or lambda. Returns the node.
    node.args.posonlyargs = self._visit_node_list(node.args.posonlyargs)
    node.args.args = self._visit_node_list(node.args.args)
    if node.args.vararg is not None:
      node.args.vararg = self.visit(node.args.vararg)
    node.args.kwonlyargs = self._visit_node_list(node.args.kwonlyargs)
    if node.args.kwarg is not None:
      node.args.kwarg = self.visit(node.args.kwarg)
    return node

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visitor for function definitions. A non-isolated scope covers decorators, return annotation,
    # argument annotations/defaults and the binding of the function name; it is attached to the
    # node. Then an isolated scope for the function holds a scope for the arguments (attached to
    # the arguments node), a scope for the body (body-scope key), and is itself recorded under
    # the args-and-body key. Returns the node.
    with self.state[_FunctionOrClass] as fn:
      fn.node = node
      # The FunctionDef node itself has a Scope object that tracks the creation
      # of its name, along with the usage of any decorator accompanying it.
      self._enter_scope(False)
      node.decorator_list = self.visit_block(node.decorator_list)
      if node.returns:
        node.returns = self._process_annotation(node.returns)
      # Argument annotartions (including defaults) affect the defining context.
      node = self._visit_arg_annotations(node)

      function_name = qual_names.QN(node.name)
      self.scope.modified.add(function_name)
      self.scope.bound.add(function_name)
      self._exit_and_record_scope(node)

      # A separate Scope tracks the actual function definition.
      self._enter_scope(True, node.name)

      # Keep a separate scope for the arguments node, which is used in the CFG.
      self._enter_scope(False, node.name)

      # Arg declarations only affect the function itself, and have no effect
      # in the defining context whatsoever.
      node = self._visit_arg_declarations(node)

      self._exit_and_record_scope(node.args)

      # Track the body separately. This is for compatibility reasons, it may not
      # be strictly needed.
      self._enter_scope(False, node.name)
      node.body = self.visit_block(node.body)
      self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)

      self._exit_and_record_scope(node, NodeAnno.ARGS_AND_BODY_SCOPE)
      return node

  def visit_Lambda(self, node):
    # [FUNCTION DOC]
    # Visitor for lambdas, structured like function definitions (annotation/default scope, an
    # isolated scope with argument scope and body scope). Ensures the body expression has a scope
    # annotation. Finally the lambda's free reads (read minus bound) are added to the enclosing
    # scope, since lambdas are assumed to be used where defined. Returns the node.
    # Lambda nodes are treated in roughly the same way as FunctionDef nodes.
    with self.state[_FunctionOrClass] as fn:
      fn.node = node
      # The Lambda node itself has a Scope object that tracks the creation
      # of its name, along with the usage of any decorator accompanying it.
      self._enter_scope(False)
      node = self._visit_arg_annotations(node)
      self._exit_and_record_scope(node)

      # A separate Scope tracks the actual function definition.
      self._enter_scope(True)

      # Keep a separate scope for the arguments node, which is used in the CFG.
      self._enter_scope(False)
      node = self._visit_arg_declarations(node)
      self._exit_and_record_scope(node.args)

      # Track the body separately. This is for compatibility reasons, it may not
      # be strictly needed.
      # TODO(mdan): Do remove it, it's confusing.
      self._enter_scope(False)
      node.body = self.visit(node.body)

      # The lambda body can contain nodes of types normally not found as
      # statements, and may not have the SCOPE annotation needed by the CFG.
      # So we attach one if necessary.
      if not anno.hasanno(node.body, anno.Static.SCOPE):
        anno.setanno(node.body, anno.Static.SCOPE, self.scope)

      self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)

      lambda_scope = self.scope
      self._exit_and_record_scope(node, NodeAnno.ARGS_AND_BODY_SCOPE)

      # TODO(bhack:) https://github.com/tensorflow/tensorflow/issues/56089
      # remove after deprecation
      # Exception: lambdas are assumed to be used in the place where
      # they are defined. Therefore, their activity is passed on to the
      # calling statement.
      self.scope.read.update(lambda_scope.read - lambda_scope.bound)

      return node

  def visit_With(self, node):
    # [FUNCTION DOC]
    # Visitor for with statements: generic visit in a fresh scope recorded under the body-scope
    # key. Returns the node.
    self._enter_scope(False)
    node = self.generic_visit(node)
    self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)
    return node

  def visit_withitem(self, node):
    # [FUNCTION DOC]
    # Visitor for with items: processed as a simple statement with its own scope.
    return self._process_statement(node)

  def visit_If(self, node):
    # [FUNCTION DOC]
    # Visitor for if statements: the test is visited in a scope recorded on the test node and as
    # the condition scope of the statement; body and else branches are processed as parallel
    # blocks (body and else-branch scope keys). Returns the node.
    self._enter_scope(False)
    node.test = self.visit(node.test)
    node_scope = self._exit_and_record_scope(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, node_scope)

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_For(self, node):
    # [FUNCTION DOC]
    # Visitor for for-loops: visits target and iterable in a scope recorded on the iterable node;
    # then visits the target again (plus any extra loop test annotation) in an iterate scope;
    # body and else branches are processed as parallel blocks. Returns the node.
    self._enter_scope(False)
    node.target = self.visit(node.target)
    node.iter = self.visit(node.iter)
    self._exit_and_record_scope(node.iter)

    self._enter_scope(False)
    self.visit(node.target)
    if anno.hasanno(node, anno.Basic.EXTRA_LOOP_TEST):
      self._process_statement(anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST))
    self._exit_and_record_scope(node, tag=NodeAnno.ITERATE_SCOPE)

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_While(self, node):
    # [FUNCTION DOC]
    # Visitor for while loops: test visited in a scope recorded on the test node and as the
    # condition scope; body and else branches processed as parallel blocks. Returns the node.
    self._enter_scope(False)
    node.test = self.visit(node.test)
    node_scope = self._exit_and_record_scope(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, node_scope)

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_ExceptHandler(self, node):
    # [FUNCTION DOC]
    # Visitor for except handlers: generic visit in a fresh scope, where the exception variable
    # name is an isolated name so it does not leak. Scope is frozen but not recorded. Returns node.
    self._enter_scope(False)
    # try/except oddity: as expected, it leaks any names you defined inside the
    # except block, but not the name of the exception variable.
    if node.name is not None:
      self.scope.isolated_names.add(anno.getanno(node.name, anno.Basic.QN))
    node = self.generic_visit(node)
    self._exit_scope()
    return node


def resolve(node, context, parent_scope=None):
  # [FUNCTION DOC]
  # Entry point: runs the activity analyzer over a syntax tree, annotating nodes with scopes.
  # Inputs: the tree node, the transformer context, optional parent scope. Returns the annotated
  # node. Mutates the tree.
  return ActivityAnalyzer(context, parent_scope).visit(node)
