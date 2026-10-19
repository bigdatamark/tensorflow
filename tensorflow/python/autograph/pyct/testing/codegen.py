# [MODULE DOC]
# HELPER: test-support module for fuzzing/testing autograph's source transformations.
# Generates random, syntactically valid function definition syntax trees (gast nodes) made of
# assignments, if and while statements, comparisons, unary and binary operations and variable
# names. Randomness comes from weighted samplers (small classes holding a mapping from node kinds
# to relative weights) driven by numpy and the random module, so output is non-deterministic unless
# the caller seeds those generators. Exposes the sampler classes, two size-limit constants, the
# CodeGenerator class and a convenience function that makes one random function definition.
# No import-time side effects besides defining these names.
#
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
"""Random code generation for testing/fuzzing."""
# pylint: disable=invalid-name
import random
import string

import gast
import numpy as np

from tensorflow.python.autograph.pyct import templates


class NodeSampler(object):
  # [CLASS DOC]
  # HELPER: base class for the weighted random choosers used by CodeGenerator.
  # Subclasses set the class attribute sample_map, a mapping from choices (node classes or
  # strings) to non-negative relative weights. The base value is nothing (null), so the base
  # class itself cannot be sampled.
  sample_map = None

  def sample(self):
    # [FUNCTION DOC]
    # Draws one key of the sample map at random, with probability proportional to its weight
    # (weights are normalized by their sum). Takes no arguments, returns the chosen key.
    # Uses numpy global random state. Fails if the map is null or all weights are zero.
    nodes, magnitudes = zip(*self.sample_map.items())
    return np.random.choice(
        nodes, p=np.array(magnitudes, dtype='float32') / np.sum(magnitudes))


class StatementSampler(NodeSampler):
  # [CLASS DOC]
  # HELPER: sampler choosing the kind of statement to generate. Weights: assignment 10, print 1,
  # if 2, while 2, for 0 (never chosen).
  sample_map = dict((
      (gast.Assign, 10),
      (gast.Print, 1),
      (gast.If, 2),
      (gast.While, 2),
      (gast.For, 0),
  ))


class ExpressionSampler(NodeSampler):
  # [CLASS DOC]
  # HELPER: sampler choosing the kind of expression to generate. Weights: unary operation 1,
  # binary operation 8, name 1, call 0 (never chosen).
  sample_map = dict((
      (gast.UnaryOp, 1),
      (gast.BinOp, 8),
      (gast.Name, 1),
      (gast.Call, 0),
  ))


class CompareSampler(NodeSampler):
  # [CLASS DOC]
  # HELPER: sampler choosing a comparison operator, uniformly among equal, not equal, less,
  # less or equal, greater, greater or equal, identity and negated identity.
  sample_map = dict((
      (gast.Eq, 1),
      (gast.NotEq, 1),
      (gast.Lt, 1),
      (gast.LtE, 1),
      (gast.Gt, 1),
      (gast.GtE, 1),
      (gast.Is, 1),
      (gast.IsNot, 1),
  ))


class BinaryOpSampler(NodeSampler):
  # [CLASS DOC]
  # HELPER: sampler choosing a binary arithmetic operator, uniformly among add, subtract,
  # multiply, divide, floor divide, modulo and power.
  sample_map = dict((
      (gast.Add, 1),
      (gast.Sub, 1),
      (gast.Mult, 1),
      (gast.Div, 1),
      (gast.FloorDiv, 1),
      (gast.Mod, 1),
      (gast.Pow, 1),
  ))


class UnaryOpSampler(NodeSampler):
  # [CLASS DOC]
  # HELPER: sampler choosing a unary operator: negation (weight 1) or unary plus (weight 0,
  # never chosen).
  sample_map = dict(((gast.USub, 1), (gast.UAdd, 0)))


class NameSampler(NodeSampler):
  # [CLASS DOC]
  # HELPER: sampler choosing between the strings for a new or an existing variable name, equal
  # weights. KNOWN ISSUE: not used anywhere in this module; names are generated randomly instead.
  sample_map = dict((
      ('new', 1),
      ('existing', 1),
  ))


N_CONTROLFLOW_STATEMENTS = 10
N_FUNCTIONDEF_STATEMENTS = 10


class CodeGenerator(object):
  # [CLASS DOC]
  # Generator of random function definition trees. Holds max_depth (deepest allowed nesting of
  # control-flow statements) and depth (current nesting counter). Statement and expression kinds
  # are chosen with the sampler classes and produced by methods named after the node kind;
  # generate_statement dispatches on the chosen kind by method name. Create one with optional
  # limits and call generate_FunctionDef, or use the module-level convenience function.
  """Generate random syntactically-valid Python ASTs."""

  def __init__(self, max_depth=3, depth=0):
    # [FUNCTION DOC]
    # Stores the limits. max_depth: integer, deepest nesting of compound statements allowed
    # (default 3). depth: integer starting nesting level (default 0). Returns nothing.
    self.max_depth = max_depth
    self.depth = depth

  def generate_statement(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when a compound statement is drawn beyond the depth limit, the method
    # recurses and returns without undoing its own depth increment, so the depth counter drifts
    # upward on that path. Also, print and for statements dispatch to generators that rely on
    # node kinds that may not exist in the installed gast version (print) or are not implemented
    # at all (for has weight zero, so is never drawn).
    # Draws a statement kind with the statement sampler, increments the depth counter, redraws
    # (recursively) if the kind is while, for or if and depth exceeds the maximum, otherwise calls
    # the generator method named after the kind, decrements depth and returns the new node.
    # Takes no arguments; returns a syntax tree statement node.
    """Generate a statement node, dispatching to the correct class method."""
    desired_node = StatementSampler().sample()
    self.depth += 1

    # Enforce some constraints on generating statements.
    # E.g., if statements need at least 3 readable variables.
    # If we fail to satisfy our constraints, draw another sample.
    if desired_node in (gast.While, gast.For, gast.If):
      if self.depth > self.max_depth:
        return self.generate_statement()

    # Go get the generator method and run it
    method = 'generate_' + desired_node.__name__
    visitor = getattr(self, method)
    node = visitor()
    self.depth -= 1
    return node

  def sample_node_list(self, low, high, generator):
    # [FUNCTION DOC]
    # HELPER: builds a list of generated nodes for bodies and argument lists.
    # Calls generator (a callback with no arguments returning a node) a random number of times:
    # the count is drawn uniformly from low (inclusive) to high (exclusive). Returns the list.
    # Raises an error if low is not less than high. Uses numpy global random state.
    """Generate a list of statements of random length.

    Args:
      low: Fewest number of statements to generate.
      high: Highest number of statements to generate.
      generator: Function to call to generate nodes.

    Returns:
      A list of statements.
    """
    statements = []
    for _ in range(np.random.randint(low, high)):
      statements.append(generator())
    return statements

  def generate_Name(self, ctx=gast.Load()):
    # [FUNCTION DOC]
    # Generates a name node with a random identifier: an underscore followed by four random
    # lowercase letters (collisions between calls are possible). ctx is the usage context node
    # (load by default, store or parameter marker for targets and arguments). The default context
    # object is created once and shared across calls. Returns the name node, without annotation.
    variable_name = '_' + ''.join(
        random.choice(string.ascii_lowercase) for _ in range(4))
    return gast.Name(variable_name, ctx=ctx, annotation=None)

  def generate_BinOp(self):
    # [FUNCTION DOC]
    # Generates a binary operation node with a randomly chosen arithmetic operator whose two
    # operands are freshly generated random names (not nested expressions). Returns the node.
    # TODO(alexbw): convert to generate_expression when we get to limit
    # expression depth.
    op = BinaryOpSampler().sample()()
    return gast.BinOp(self.generate_Name(), op, self.generate_Name())

  def generate_Compare(self):
    # [FUNCTION DOC]
    # Generates a comparison node: a random name, one randomly chosen comparison operator and
    # another random name. Returns the node. Used as the test of if and while statements.
    op = CompareSampler().sample()()
    return gast.Compare(self.generate_Name(), [op], [self.generate_Name()])

  def generate_UnaryOp(self):
    # [FUNCTION DOC]
    # Generates a unary operation node applying a randomly chosen unary operator (in practice
    # negation) to a random name. Returns the node.
    operand = self.generate_Name()
    op = UnaryOpSampler().sample()()
    return gast.UnaryOp(op, operand)

  def generate_expression(self):
    # [FUNCTION DOC]
    # Draws an expression kind with the expression sampler and calls the generator method named
    # after it (unary operation, binary operation or name; call has weight zero). Returns the
    # new expression node. Takes no arguments.
    desired_node = ExpressionSampler().sample()
    # Go get the generator method and run it
    method = 'generate_' + desired_node.__name__
    generator = getattr(self, method)
    return generator()

  def generate_Assign(self):
    # [FUNCTION DOC]
    # Generates an assignment node: a random name in store context as the single target and a
    # randomly generated expression as the value. Returns the node.
    """Generate an Assign node."""
    # Generate left-hand side
    target_node = self.generate_Name(gast.Store())
    # Generate right-hand side
    value_node = self.generate_expression()
    # Put it all together
    node = gast.Assign(targets=[target_node], value=value_node)
    return node

  def generate_If(self):
    # [FUNCTION DOC]
    # Generates an if node with a random comparison as test, and random non-empty bodies for the
    # true and false branches, each with 1 to 4 random statements (nested statements may recurse
    # through generate_statement). Returns the node.
    """Generate an If node."""
    test = self.generate_Compare()

    # Generate true branch statements
    body = self.sample_node_list(
        low=1,
        high=N_CONTROLFLOW_STATEMENTS // 2,
        generator=self.generate_statement)

    # Generate false branch statements
    orelse = self.sample_node_list(
        low=1,
        high=N_CONTROLFLOW_STATEMENTS // 2,
        generator=self.generate_statement)

    node = gast.If(test, body, orelse)
    return node

  def generate_While(self):
    # [FUNCTION DOC]
    # Generates a while node with a random comparison as test, a body of 1 to 9 random statements
    # and an empty else clause. Returns the node.
    """Generate a While node."""

    test = self.generate_Compare()
    body = self.sample_node_list(
        low=1, high=N_CONTROLFLOW_STATEMENTS, generator=self.generate_statement)
    orelse = []  # not generating else statements

    node = gast.While(test, body, orelse)
    return node

  def generate_Call(self):
    # [FUNCTION DOC]
    # Placeholder: call generation is not implemented. Always raises a not-implemented error
    # (the expression sampler gives calls weight zero so it is never reached normally).
    raise NotImplementedError

  def generate_Return(self):
    # [FUNCTION DOC]
    # Generates a return node whose value is a random expression. Returns the node.
    return gast.Return(self.generate_expression())

  def generate_Print(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the statement sampler and the method name refer to a print statement node
    # kind, but print is a function call in current Python; this builds a call through the
    # template utility instead and returns its first statement.
    # Generates a statement printing a random expression via a code template. Returns the node.
    return templates.replace('print(x)', x=self.generate_expression())[0]

  def generate_FunctionDef(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: builds argument and function definition nodes using an older positional gast
    # constructor layout and a parameter-context marker; may fail on gast versions with a
    # different node signature.
    # Generates a function definition: a random name, 2 to 9 random parameter names, and a body of
    # 1 to 9 random statements followed by a final return of a random expression. No decorators
    # and no return annotation. Returns the function definition node.
    """Generate a FunctionDef node."""

    # Generate the arguments, register them as available
    arg_vars = self.sample_node_list(
        low=2, high=10, generator=lambda: self.generate_Name(gast.Param()))
    args = gast.arguments(arg_vars, None, [], [], None, [])

    # Generate the function body
    body = self.sample_node_list(
        low=1, high=N_FUNCTIONDEF_STATEMENTS, generator=self.generate_statement)
    body.append(self.generate_Return())
    fn_name = self.generate_Name().id
    node = gast.FunctionDef(fn_name, args, body, (), None)
    return node


def generate_random_functiondef():
  # [FUNCTION DOC]
  # Convenience entry point: creates a code generator with default limits and returns one random
  # function definition syntax tree. Takes no arguments.
  return CodeGenerator().generate_FunctionDef()
