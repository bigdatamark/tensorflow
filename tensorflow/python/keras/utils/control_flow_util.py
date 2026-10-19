# [MODULE DOC]
# HELPER: helper module of control-flow utilities used by Keras.
# Local copy of a few functions from the core control flow utilities: queries about whether a
# graph or op sits inside XLA or while-loop contexts, plus smart_cond and constant_value helpers
# that pick between callables using a possibly-constant predicate. No import-time side effects.
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
"""Utility functions for control flow.

This file is copied from tensorflow/python/ops/control_flow_util.py.
"""

from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables


def InXlaContext(graph):
  # [FUNCTION DOC]
  # HELPER: control-flow query used by GraphOrParentsInXlaContext and Keras layers/backend code.
  # Reports whether the given graph's current control flow context is, or is nested inside, an
  # XLA context. Input: graph, a TensorFlow graph object. Returns True if an enclosing XLA context
  # exists, otherwise False. Reads a private attribute of the graph; no side effects. Raises
  # whatever the graph raises if it lacks the control flow accessor.
  ctxt = graph._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingXLAContext(ctxt) is not None


def GraphOrParentsInXlaContext(graph):
  # [FUNCTION DOC]
  # HELPER: control-flow query for code that may run in a nested function graph.
  # Walks from the given graph outward through its chain of outer graphs and returns True as soon
  # as any of them is inside an XLA context. Returns False when a graph has no outer graph
  # attribute (the walk ends at the outermost graph). Input: graph, a TensorFlow graph. No side
  # effects.
  while True:
    if InXlaContext(graph): return True
    try:
      graph = graph.outer_graph
    except AttributeError:
      return False


def IsInWhileLoop(op):
  # [FUNCTION DOC]
  # HELPER: control-flow query for Keras code.
  # Reports whether the given graph operation was created inside a while loop, by looking up its
  # control flow context and searching its ancestors for a while context. Input: op, a graph
  # operation. Returns True or False. Reads a private attribute of the op; no side effects.
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingWhileContext(ctxt) is not None


def GetContainingWhileContext(ctxt, stop_ctxt=None):
  # [FUNCTION DOC]
  # HELPER: context-walking utility for IsInWhileLoop.
  # Walks from the given control flow context outward through its outer contexts and returns the
  # first one that is a while context, or that equals the optional stop context. Inputs: ctxt, a
  # control flow context (may be nothing); stop_ctxt, optional context at which the search ends
  # early. Returns the matching context, or nothing if the chain is exhausted. No side effects.
  """Returns the first ancestor WhileContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a WhileContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext
    stop_ctxt: ControlFlowContext, optional. If provided, the search will end
      if it sees stop_ctxt.

  Returns:
    `ctxt` if `ctxt` is a WhileContext, the most nested WhileContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.  If `stop_ctxt` is not
    `None`, this returns `ctxt` if it matches `stop_ctxt` in its traversal.
  """
  while ctxt:
    if ctxt.IsWhileContext() or ctxt == stop_ctxt: return ctxt
    ctxt = ctxt.outer_context
  return None


def GetContainingXLAContext(ctxt):
  # [FUNCTION DOC]
  # HELPER: context-walking utility for InXlaContext.
  # KNOWN ISSUE: the docstring says it returns nothing when the context is "not in a while loop";
  # the code actually checks for XLA contexts, so that wording is a copy-paste error.
  # Walks from the given control flow context outward through outer contexts and returns the first
  # one that is an XLA context, or nothing if none is found. Input: ctxt, a control flow context
  # (may be nothing). No side effects.
  """Returns the first ancestor XLAContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a XLAContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a XLAContext, the most nested XLAContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.
  """
  while ctxt:
    if ctxt.IsXLAContext(): return ctxt
    ctxt = ctxt.outer_context
  return None


def smart_cond(pred, true_fn=None, false_fn=None, name=None):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: conditional-execution wrapper used by Keras layers (for example training flags).
  # Chooses between two callables. If pred is a variable, a dynamic graph conditional is always
  # used, because its value is not known statically. Otherwise the framework smart conditional
  # runs: a constant predicate runs only the chosen callable, a tensor predicate becomes a graph
  # conditional. Inputs: pred, scalar boolean-like value; true_fn and false_fn, callables run when
  # pred is true or false; name, optional name prefix. Returns the result of the chosen callable.
  # Raises a type error if a callable is not callable (from the underlying functions).
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if isinstance(pred, variables.Variable):
    return cond.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  return smart_module.smart_cond(
      pred, true_fn=true_fn, false_fn=false_fn, name=name)


def constant_value(pred):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: static-value extraction for smart conditionals in Keras.
  # Tries to determine the boolean value of pred without running a graph. A tensor yields its
  # statically known value, or nothing if unknown. The integers 0 and 1 (and booleans, since they
  # compare equal to them) become booleans. A variable yields nothing. Anything else raises a
  # type error. Input: pred, scalar. Returns True, False or nothing. No side effects.
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Args:
    pred: A scalar, either a Python bool or a TensorFlow boolean variable
      or tensor, or the Python integer 1 or 0.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Variable, Tensor or bool, or Python
      integer 1 or 0.
  """
  if isinstance(pred, tensor.Tensor):
    return tensor_util.constant_value(pred)
  if pred in {0, 1}:  # Accept 1/0 as valid boolean values
    return bool(pred)
  if isinstance(pred, bool):
    return pred
  if isinstance(pred, variables.Variable):
    return None
  raise TypeError("`pred` must be a Tensor, or a Python bool, or 1 or 0. "
                  "Found instead: %s" % type(pred))
