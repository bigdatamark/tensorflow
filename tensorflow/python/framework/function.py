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
# =============================================================================
# [MODULE DOC]
# Legacy graph-mode function support (the Defun decorator). Provides the decorator, defined and
# overloaded function classes, an internal function-building graph that captures outside tensors
# or variables, helpers to trace a python callable into a graph, to add call ops, and to rebuild
# functions from function library protos, plus getters for captured variables, inputs and
# arguments. Marked experimental and obsolete. Import-time state: a flag marking the open-source
# build and a table mapping data types to short codes.
"""Python front-end supports for functions.

NOTE: At this time, functions are experimental and subject to change!. Proceed
with caution.
"""

import collections
import hashlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect

is_oss = True  # updated by copybara


# TODO(b/136040013): Drop support for Defun.
class Defun(object):
  # [CLASS DOC]
  # Obsolete decorator (superseded by the modern function tracing API) that turns a python
  # function building graph ops into a graph function. Constructed with argument dtypes and
  # options (name, gradient functions, output names, extra attributes); applying it to a callable
  # returns a defined function, or an overloaded function when argument types are unknown.
  """Obsolete. Slated for deletion. Please use tf.function instead.

  Known feature gaps while migrating to tf.function (could be outdated):
  - tf.function doesn’t support Send/Recv capability since it doesn’t share
    rendezvous with the main graph but always creates a new one.
  - tf.function doesn’t support custom gradient function directly, instead you
    need to define the function inside a tf.custom_gradient wrapper together
    with the gradient function.
  - Unlike Defun, Keras layers used inside a tf.function need to be created only
    once to avoid variable recreation.
  - Defun respects the device assignments and applies them to the function body
    but tf.function needs it to be done manually.
  - Defun might prune out unused ops automatically but tf.function doesn't.

  Limitations of Defun:
  - Original source locations are not preserved so errors do not include
    full/valid stack traces.
  - Only supports linear sequence of arguments and return values, putting the
    burden on the caller to pack/unpack everything across a Defun boundary into
    tuples (as opposed to passing list and dict-like structures directly).
  - Does not support overloading or late-bound specializations.
  - Has its own way for defining gradient overrides which does not follow
    current conventions.
  - Cannot support imperative control flow or automatic control dependencies.
  - Does not reflect statefulness in the graph and has a calling convention that
    differs from how more modern tools interact.
  - Is only compatible with graph building mode.

  Decorator used to define TensorFlow functions.

  Use this decorator to make a Python function usable directly as a TensorFlow
  function.

  The decorated function must add ops to the default graph and return zero or
  more `Tensor` objects.  Call the decorator with named arguments, one for each
  argument of the function to decorate, with the expected type of the argument
  as value.

  For example if the function to decorate accepts two `tf.float32` arguments
  named `x` and `y`, call the decorator with:

      @Defun(tf.float32, tf.float32)
      def foo(x, y):
        ...

  When you call the decorated function, it adds the `call` ops to the
  default graph. In addition, it adds the definition of the function into the
  default graph. Because the addition of the function into the graph
  is deferred, the decorator can be used anywhere in the program.

  Any variables created inside of the function are hoisted into the outer graph.
  Note that the variables are created in the variable scope that was active
  during the first call to the function. Subsequent function calls will refer to
  the same set of variables.

  Definitions of functions in a graph are frozen as soon as the graph is used to
  create a session. However, new functions and new calls to existing functions
  may be added to the graph, with the new functions themselves becoming
  immediately frozen.

  Example, but also see the [How To on functions](link_needed).

  ```python
  # Defining the function.
  @tf.Defun(tf.float32, tf.float32)
  def MyFunc(x, y):
    return x + y, x - y

  # Building the graph.
  a = tf.constant([1.0])
  b = tf.constant([2.0])
  c, d = MyFunc(a, b, name='mycall')
  ```
  """

  def __init__(self, *input_types, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says at most one of the two gradient options may be given, but
    # nothing here enforces that; also the documented shape function option stays in the leftover
    # keyword options rather than being popped.
    # Constructor of the decorator. Inputs: positional dtypes of the function arguments (may be
    # empty); keyword options. Pops function name, gradient function, python gradient function and
    # output names; the remaining keyword options are kept and later forwarded to the defined
    # function and its call sites. Returns nothing.
    """Create a `Defun` decorator.

    Args:
      *input_types: A list of `tf.DType`
      **kwargs: Optional keyword arguments, including
         func_name - (optional).  A python string, the name to use to
           declare this `Function` in the graph.

         grad_func - (optional).  A function implementing the gradient
           of the function-to-register.  This is must be a
           `_DefinedFunction` object. The gradient
           function must satisfy the criterion defined in
           function.proto:GradientDef.

         python_grad_func - (optional).  A function implementing the
           gradient of the function python-side. This function must
           take the current op and the gradients w.r.t. its outputs,
           and return the gradients w.r.t. the inputs. That is it must
           implement the interface expected by `tf.RegisterGradient`).
           This will be called by tf.gradients to add the gradient ops
           to the graph. At most one of grad_func and python_grad_func
           can be specified.

         out_names = (optional). A list of strings, one per output
           tensor.

         shape_func - (optional). A function taking the op and returning a list
           of static shapes to set for the function's outputs.
    """
    self._input_types = input_types
    self._func_name = kwargs.pop("func_name", None)
    self._grad_func = kwargs.pop("grad_func", None)
    self._python_grad_func = kwargs.pop("python_grad_func", None)
    self._out_names = kwargs.pop("out_names", None)
    self._extra_kwargs = kwargs

  def __call__(self, func):
    # [FUNCTION DOC]
    # Applies the decorator to a callable. Input: func must be callable (else value error) and must
    # not have keyword-argument catch-alls or default values (value error). Counts allowed
    # arguments (variable positional arguments allow many; the first argument of a bound method is
    # excluded). If input types were given, their count must fit the allowed range (value error
    # otherwise) and a defined function is returned. If none were given and the callable takes no
    # arguments, a defined function with no inputs is returned. Otherwise an overloaded function is
    # returned whose definition is deferred until called. Name, gradients, output names and extra
    # options from the decorator are forwarded.
    # Various sanity checks on the callable func.
    if not callable(func):
      raise ValueError(f"Function {func} must be a callable.")

    # Func should not use kwargs and defaults.
    argspec = tf_inspect.getargspec(func)
    if argspec.keywords or argspec.defaults:
      raise ValueError(
          "Functions with argument defaults or keywords arguments are not "
          f"supported. {func} has defaults {argspec.defaults} and keywords "
          f"{argspec.keywords}.")

    # Computes how many arguments 'func' has.
    min_args = len(argspec.args)
    max_args = min_args
    if argspec.varargs:
      max_args = 1000000
    argnames = argspec.args
    if tf_inspect.ismethod(func):
      # 1st argument is the "class" type.
      min_args -= 1
      argnames = argnames[1:]

    if self._input_types:
      # If Defun is given a list of types for the inputs, the number
      # of input types should be compatible with 'func'.
      num = len(self._input_types)
      if num < min_args or num > max_args:
        raise ValueError(
            "The number of tf.function input types is not compatible with the "
            f"allowed arguments of {func}. The tf.function have {num} input "
            f"types, while the python function allows minimum {min_args} and "
            f"maximum {max_args} arguments.")
      return _DefinedFunction(
          func,
          argnames,
          self._input_types,
          self._func_name,
          self._grad_func,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)

    # 'func' expects no arguments and input types is an empty list.
    if min_args == 0 and max_args == 0:
      return _DefinedFunction(
          func, [], [],
          self._func_name,
          self._grad_func,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)

    # Input types are unknown. It's an overloaded function and hence
    # its definition needs to be deferred until it's called.
    return _OverloadedFunction(
        func,
        argnames,
        self._func_name,
        self._grad_func,
        self._python_grad_func,
        out_names=self._out_names,
        **self._extra_kwargs)


class _DefinedFunctionDeleter(object):
  # [CLASS DOC]
  # HELPER: tiny owner object whose finalizer unregisters a function from the eager context, so a
  # defined function's eager registration is cleaned up when it is garbage collected. Holds only
  # the function name.
  """Unregister function from eager context."""

  __slots__ = ["name"]

  def __init__(self, name):
    # [FUNCTION DOC]
    # Constructor storing the name of the function to unregister later.
    self.name = name

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: removes the named function from the eager context. Swallows type and attribute
    # errors that happen at interpreter or module teardown. Returns nothing.
    try:
      context.remove_function(self.name)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the context module
      # already being unloaded, self._handle._handle_data no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
      # partially unloaded.
    except AttributeError:
      pass  # 'NoneType' object has no attribute 'eager_mode' when context has
      # been unloaded. Will catch other module unloads as well.


class _DefinedFunction(object):
  # [CLASS DOC]
  # Legacy Defun support: a function with known argument types whose definition is built lazily
  # by tracing the python callable. Holds the callable, names/types, gradient settings, extra call
  # options, the C-level function, cached op definition, captured inputs and stateful ops. Calling
  # an instance adds the definition to the default graph and creates a call op.
  """_DefinedFunction encapsulates a function definition and its properties.

  Attributes:
    name: The function name.
    definition: The definition of this function. A FunctionDef proto.
    cached_definition: Same as definition. Needed to match AtomicFunction API.
    grad_func_name: If not None, the name of this function's gradient function.
    python_grad_func: A python callable implementing the gradient of
      the function python-side.
  """

  def __init__(self,
               func,
               argnames,
               input_types,
               func_name=None,
               grad_func=None,
               python_grad_func=None,
               out_names=None,
               shape_func=None,
               capture_by_value=False,
               allowlisted_stateful_ops=None,
               capture_resource_var_by_value=True,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the documented ValueError for invalid definitions is not raised here; a bad
    # input_types value (not a list or tuple) triggers an assertion failure instead.
    # Constructor. Inputs: func callable; argnames list of names; input_types list or tuple of
    # dtypes; optional func_name, grad_func, python_grad_func, out_names, shape_func; flags for
    # capturing by value and for resource variables (default true); allowlisted stateful ops
    # (default empty set); extra keyword options passed to each call site. Records the innermost
    # device function of the default graph, leaves the definition unbuilt (lazy), and generates
    # argument names, falling back to generic names for missing ones.
    """Creates _DefinedFunction.

    Args:
      func:  A python callable which constructs a tf function body.
      argnames: A list of strings for function argument names.
      input_types: The function's argument types. Can be a tuple, list of
        tf data types.
      func_name: The function name. Defaults to None, in which derives from
        'func'.
      grad_func: This function's gradient function, if not None. Defaults
        to None.
      python_grad_func: A python callable implementing the gradient of
        the function python-side.
      out_names: An optional list of strings for the function return value
        names.
      shape_func: An optional function mapping an op to a list of static
        output shapes.
      capture_by_value: Boolean (defaults to False). If True, captured values
        will be copied into the function body.
      allowlisted_stateful_ops: A set of ops that if stateful we ignore and
        copy into the function body, when `capture_by_value` is True.
      capture_resource_var_by_value: Boolean (defaults to True). If False,
        captured resource variable returns the handle instead of value.
      **kwargs: The keyword arguments. **kwargs is passed to every call
        site of this function.

    Raises:
      ValueError: The function definition is invalid.

    """
    self._func = func
    self._input_types = input_types
    self._func_name = func_name
    self._grad_func = grad_func
    self._python_grad_func = python_grad_func
    self._out_names = out_names
    self._shape_func = shape_func
    self._capture_by_value = capture_by_value
    self._allowlisted_stateful_ops = allowlisted_stateful_ops
    if self._allowlisted_stateful_ops is None:
      self._allowlisted_stateful_ops = set()
    self._capture_resource_var_by_value = capture_resource_var_by_value
    self._extra_kwargs = kwargs
    # Constructed only when C API is disabled, lazily
    self._definition = None
    # Constructed only when C API is enabled, lazily
    self._c_func = None
    self._function_deleter = None
    self._sub_functions = {}  # Constructed with _definition or _c_func
    # pylint: disable=protected-access
    device_funcs = ops.get_default_graph()._device_functions_outer_to_inner
    # pylint: enable=protected-access

    # Get the innermost device if possible.
    self._caller_device = device_funcs[-1] if device_funcs else None

    # Cached OpDef for this function. When C API is enabled, this is
    # the only part of FunctionDef that we cache in Python. When C API
    # is disabled the whole _definition is available and this is simply
    # another reference to _definition.signature
    self._op_def = None

    assert isinstance(input_types, (list, tuple))
    self._arg_types = input_types
    self._arg_names = [argnames[i] if i < len(argnames) else ("arg%d" % i)
                       for i in range(len(input_types))]

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the function name, building the definition first if needed (the name may
    # be derived from a hash of the body).
    """Function name."""
    self._create_definition_if_needed()
    return self._func_name

  @property
  def cached_definition(self):
    # [FUNCTION DOC]
    # HELPER: property alias of the definition property, existing to match the atomic function
    # interface.
    # KNOWN ISSUE: despite the name it is not cached here; it recomputes each time.
    return self.definition

  @property
  def definition(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: every access with a C function re-serializes the definition, and in eager mode
    # re-registers it with the eager context and replaces the deleter object, so it is not a cheap
    # or idempotent property read.
    # Property returning the function definition proto. Builds it if needed; for a C-level function
    # converts it to a new proto each time, and in eager mode registers it in the eager context
    # with a deleter that unregisters it on cleanup. Otherwise returns the stored definition.
    """Function definition proto."""
    self._create_definition_if_needed()
    if self._c_func:
      with c_api_util.tf_buffer() as buf:
        with self._c_func.get() as func:
          c_api.TF_FunctionToFunctionDef(func, buf)
          fdef = function_pb2.FunctionDef()
          proto_data = c_api.TF_GetBuffer(buf)
          fdef.ParseFromString(compat.as_bytes(proto_data))
          with ops.init_scope():
            if context.executing_eagerly():
              context.add_c_function(func)
              self._function_deleter = _DefinedFunctionDeleter(
                  fdef.signature.name)
      return fdef
    return self._definition

  @property
  def _signature(self):
    # [FUNCTION DOC]
    # HELPER: private property returning the cached op definition (signature) after making sure
    # the definition has been built.
    self._create_definition_if_needed()
    return self._op_def

  def set_grad_func(self, grad_func):
    # [FUNCTION DOC]
    # Sets the gradient function. Input: grad_func must be a defined-function object. Uses
    # assertions: fails if a gradient is already set or the argument has the wrong type.
    # Returns nothing; mutates this object.
    """Specifies the gradient function of this function."""
    assert not self._grad_func
    assert isinstance(grad_func, _DefinedFunction)
    self._grad_func = grad_func

  @property
  def grad_func_name(self):
    # [FUNCTION DOC]
    # Property returning the name of the gradient function (forcing its definition to be built),
    # or nothing if there is no gradient function.
    """Returns the name of the gradient function."""
    return self._grad_func.name if self._grad_func else None

  @property
  def python_grad_func(self):
    # [FUNCTION DOC]
    # Property returning the python-side gradient callable, or nothing if none was given.
    """Python gradient function callable."""
    return self._python_grad_func

  @property
  def declared_input_types(self):
    # [FUNCTION DOC]
    # Property returning the explicitly declared input data types given at construction.
    """Returns the list of data types of explicit declared inputs."""
    return self._input_types

  @property
  def captured_inputs(self):
    # [FUNCTION DOC]
    # Property: builds the definition if needed and returns the list of outside tensors that the
    # body captured implicitly (to be passed as extra call arguments).
    """Returns the list of implicitly captured inputs."""
    self._create_definition_if_needed()
    return self._extra_inputs

  @property
  def stateful_ops(self):
    # [FUNCTION DOC]
    # Property: builds the definition if needed and returns the list of (name, type) pairs of
    # stateful operations in the function body.
    """Returns the list of stateful ops in function definition.

    Returns:
      A list of (op.name, op.type) pairs.
    """
    self._create_definition_if_needed()
    return self._stateful_ops

  def _create_definition_if_needed(self):
    # [FUNCTION DOC]
    # HELPER: ensures the function definition exists, building it in graph mode (so tracing is not
    # affected by eager execution). Returns nothing; may build and cache the definition.
    """Creates the function definition if it's not created yet."""
    with context.graph_mode():
      self._create_definition_if_needed_impl()

  def _create_definition_if_needed_impl(self):
    # [FUNCTION DOC]
    # HELPER: implementation of lazy definition building; call through the wrapper instead.
    # KNOWN ISSUE: the first branch (taken when the graph lacks a C graph) is dead code, as the
    # source comment admits the C API is always enabled; and the stateful-op list is set only after
    # building, so an early return leaves it as previously computed.
    # Does nothing if already built. Otherwise shares variable collections with the parent graph,
    # traces the callable into a temporary function graph, saves captured inputs and sub-functions,
    # parses extra keyword options into attributes, builds the C-level function (hash appended to
    # the name when no name was given), applies attributes, caches the op definition and resolved
    # function name, and records the stateful ops as pairs of name and type.
    """This is not what you want, see _create_definition_if_needed."""
    if self._definition is not None or self._c_func is not None:
      return

    # Copy variable collections (by reference) from the parent graph such that
    # name based variable sharing (e.g. via tf.make_template) works between the
    # func graph and parent graph.
    variable_keys = []
    variable_keys.extend(ops.GraphKeys._VARIABLE_COLLECTIONS)  # pylint: disable=protected-access
    variable_keys.append(vs._VARSTORE_KEY)  # pylint: disable=protected-access

    parent_graph = ops.get_default_graph()
    collections_ref = {
        key: parent_graph.get_collection_ref(key) for key in variable_keys}

    temp_graph = func_graph_from_py_func(
        self._func,
        self._arg_names,
        self._arg_types,
        self._func_name,
        self._capture_by_value,
        self._caller_device,
        collections_ref=collections_ref,
        allowlisted_stateful_ops=self._allowlisted_stateful_ops,
        capture_resource_var_by_value=self._capture_resource_var_by_value)

    self._extra_inputs = temp_graph.extra_inputs
    # pylint: disable=protected-access
    self._sub_functions = temp_graph._functions
    # pylint: enable=protected-access

    # Extra kwargs are treated as attrs on the function def.
    if self._func_name:
      base_func_name = self._func_name
    else:
      base_func_name = function_utils.get_func_name(self._func)
      if self._grad_func:
        base_func_name += ("_%s" % self._grad_func.name)
    kwargs_attr = _parse_kwargs_as_attrs(base_func_name, **self._extra_kwargs)

    # FIXME(feyu): C API is always enabled now. The if-true branch never runs.
    if not temp_graph._c_graph:  # pylint: disable=protected-access
      # Build the FunctionDef
      self._definition = graph_to_function_def.graph_to_function_def(
          temp_graph,
          temp_graph.get_operations(),
          temp_graph.inputs,
          temp_graph.outputs,
          out_names=self._out_names)

      for k in kwargs_attr:
        self._definition.attr[k].CopyFrom(kwargs_attr[k])

      # Hash the definition and its dependencies.
      self._hash_str = self._create_hash_str(
          self._definition.signature.input_arg,
          self._definition.signature.output_arg, self._definition.node_def)

      # Finally, we decide the function name to use.  If not specified,
      # make up something which is almost certainly unique (but deterministic).
      if not self._func_name:
        self._func_name = "_".join([base_func_name, self._hash_str])
      self._definition.signature.name = self._func_name
      if self._func.__doc__:
        self._definition.signature.description = self._func.__doc__

      self._op_def = self._definition.signature
    else:  # C API is enabled
      output_names = ([compat.as_bytes(x) for x in self._out_names]
                      if self._out_names else [])
      description = self._func.__doc__ or None
      # pylint: disable=protected-access
      with temp_graph._c_graph.get() as c_graph:
        c_func = c_api.TF_GraphToFunction_wrapper(
            c_graph,
            base_func_name,
            self._func_name is None,  # append_hash_to_fn_name
            None,  # opers
            [t._as_tf_output() for t in temp_graph.inputs],
            [t._as_tf_output() for t in temp_graph.outputs],
            output_names,
            [],  # control_outputs
            [],  # control_output_names
            None,  # opts
            description)
      self._c_func = c_api_util.ScopedTFFunction(c_func, base_func_name)
      # pylint: enable=protected-access
      self._set_c_attrs(kwargs_attr)

      # Set cached fields: _op_def and _func_name (if not already set)
      self._op_def = self.definition.signature
      if self._func_name:
        assert self._func_name == self._op_def.name
      else:
        self._func_name = compat.as_str(self._op_def.name)

    self._stateful_ops = [(op.name, op.type)
                          for op in temp_graph.get_operations()
                          if op._is_stateful]  # pylint: disable=protected-access

  def _set_c_attrs(self, attrs):
    # [FUNCTION DOC]
    # HELPER: attaches attributes to the underlying C-level function.
    # Input: attrs is a mapping from attribute names to attribute value protos. Requires the C
    # function to exist. Serializes each value and sets it on the function. Returns nothing.
    """Sets `attrs` as attributes of self._c_func.

    Requires that self._c_func is not None.

    Args:
      attrs: a dictionary from attribute name to attribute proto value
    """
    for name, attr_value in attrs.items():
      serialized = attr_value.SerializeToString()
      # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
      # It might be worth creating a convenient way to re-use the same status.
      with self._c_func.get() as func:
        c_api.TF_FunctionSetAttrValueProto(func, compat.as_str(name),
                                           serialized)

  def _create_hash_str(self, input_arg, output_arg, node_def):
    # [FUNCTION DOC]
    # HELPER: derives a deterministic name suffix for a function definition.
    # Inputs: input and output argument definitions and the node definitions of the function.
    # Feeds a SHA-1 hash with the serialized arguments, then nodes sorted by name (name, op, inputs,
    # attribute count and sorted attributes). Returns the first eight hex characters. Pure.
    """Creates an 8-character string unique to this input.

    Args:
      input_arg: the input_arg field of an OpDef
                 (e.g. self._definition.signature.input_arg)
      output_arg: the output_arg field of an OpDef
                 (e.g. self._definition.signature.output_arg)
      node_def: the node_def field of a FunctionDef
                (e.g. self._definition.node_def)

    Returns:
      The unique string for this input
    """
    hasher = hashlib.sha1()

    def update_num(n):
      # [FUNCTION DOC]
      # HELPER: feeds an integer, written in hexadecimal text, into the enclosing hash.
      hasher.update(compat.as_bytes("%x" % n))

    def update_str(s):
      # [FUNCTION DOC]
      # HELPER: feeds a length-prefixed string (or bytes) into the enclosing hash.
      update_num(len(s))
      hasher.update(compat.as_bytes(s))

    def update_strs(slist):
      # [FUNCTION DOC]
      # HELPER: feeds a count followed by each string of the list into the enclosing hash.
      update_num(len(slist))
      for s in slist:
        update_str(s)

    for adef in input_arg:
      update_str(adef.SerializeToString())

    for adef in output_arg:
      update_str(adef.SerializeToString())

    for n in sorted(node_def, key=lambda n: n.name):
      update_str(n.name)
      update_str(n.op)
      update_strs(n.input)
      update_num(len(n.attr))
      # NOTE: protobuf map serialization does not guarantee ordering.
      for k in sorted(n.attr):
        update_str(k)
        update_str(n.attr[k].SerializeToString())

    return hasher.hexdigest()[:8]

  def add_to_graph(self, g):
    # [FUNCTION DOC]
    # Registers this function in a graph. Input: g is the target graph. Ensures the definition is
    # built, then in eager mode adds the definition to the eager context, otherwise adds the
    # function to the graph (g is not used for this part in eager mode). Also recursively adds the
    # sub-functions it uses and its gradient function to g. Returns nothing.
    """Adds this function into the graph g."""
    self._create_definition_if_needed()

    # Adds this function into 'g'.
    # pylint: disable=protected-access
    if context.executing_eagerly():
      context.context().add_function_def(self.definition)
    else:
      g._add_function(self)
    # pylint: enable=protected-access

    # Ensures related sub-routines are defined in 'g', too.
    for f in self._sub_functions.values():
      g._add_function_recursive(f)  # pylint: disable=protected-access

    # Adds its gradient function, too.
    if self._grad_func:
      self._grad_func.add_to_graph(g)

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Calls the function: adds its definition to the default graph, converts positional arguments
    # to tensors and appends captured extra inputs, and creates the call op (keyword arguments such
    # as name or noinline go to the call helper). Marks the op with a hidden reference back to this
    # object (used to find the python gradient). If a shape function was supplied, applies its
    # returned shapes to the op outputs, raising a value error if the count differs from the number
    # of outputs. Returns the output tensor, tuple of tensors, or the op when no outputs.
    self.add_to_graph(ops.get_default_graph())
    args = [ops.convert_to_tensor(_) for _ in args] + self._extra_inputs
    ret, op = _call(self._signature, *args, **kwargs)

    # Set a hidden attr in 'op' so that gradients_impl can refer back
    # to this _DefinedFunction instance to access python_grad_func.
    assert isinstance(op, ops.Operation)
    setattr(op, "__defun", self)

    if self._shape_func is not None:
      shapes = self._shape_func(op)
      if len(shapes) != len(op.outputs):
        raise ValueError(f"shape_func {self._shape_func} produced "
                         f"{len(shapes):d} shapes, which does not match "
                         f"{len(op.outputs)} outputs.")
      for (t, shape) in zip(op.outputs, shapes):
        t.set_shape(shape)
    return ret


class _OverloadedFunction(object):
  # [CLASS DOC]
  # Legacy Defun support: a function whose argument types are not declared up front. Holds the
  # callable and options and a cache mapping input dtype signatures to specialized defined
  # functions, created lazily when called with concrete tensors. Created by the decorator.
  """_OverloadedFunction encapsulates an overloaded function.

  _OverloadedFunction maintains a mapping from input types to
  instantiated _DefinedFunction in self._overload.

  """

  def __init__(self,
               func,
               argnames,
               func_name=None,
               grad_func=None,
               python_grad_func=None,
               out_names=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it creates a defined function, but this constructs an
    # overloaded-function holder; and the documented ValueError is never raised here (an assertion
    # error occurs if grad_func is not an overloaded function).
    # Constructor storing the callable, argument names, optional function name, optional gradient
    # (must be another overloaded function), python gradient callable, output names and extra
    # keyword options passed to every call site, plus an empty cache of specializations by dtypes.
    """Creates _DefinedFunction.

    Args:
      func:  A python callable which constructs a tf function body.
      argnames: A list of strings for function argument names.
      func_name: The function name. Defaults to None, in which derives from
        'func'.
      grad_func: This function's gradient function, if not None. Defaults
        to None.
      python_grad_func: A python callable implementing the gradient of
        the function python-side.
      out_names: A list of strings for the function return value names.
      **kwargs: The keyword arguments. **kwargs is passed to every call
        site of this function.

    Raises:
      ValueError: The function definition is invalid.

    """
    self._func = func
    self._argnames = argnames
    self._func_name = func_name
    assert grad_func is None or isinstance(grad_func, _OverloadedFunction)
    self._grad_func = grad_func
    self._python_grad_func = python_grad_func
    self._out_names = out_names
    self._extra_kwargs = kwargs
    self._overload = {}

  def instantiate(self, input_types):
    # [FUNCTION DOC]
    # Returns the defined function specialized for the given input dtypes, creating and caching it
    # on first use. Input: input_types is a list of dtypes. The cache key is built from the dtype
    # codes; the function name gets the key appended when a name was provided. The definition is
    # fully built immediately, and a gradient overload (if any) is instantiated with input types
    # followed by the output types. Raises a value error for unsupported dtypes. Mutates the
    # internal cache.
    """Instantiate this function given input argument types.

    Args:
      input_types: A list of data types for the inputs.

    Returns:
      _DefinedFunction for the given input types.

    """
    # Stringify the type list.
    key = _type_list_to_str(input_types)
    defined = self._overload.get(key)
    if not defined:
      # If not defined yet, define the function given the input types.
      name = self._func_name
      if name is not None:
        name = "_".join([name, key])
      defined = _DefinedFunction(
          self._func,
          self._argnames,
          input_types,
          name,
          None,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)
      _ = defined.name  # Fully instantiate the function definition.
      if self._grad_func:
        # If _grad_func is given, it is another
        # _OverloadedFunction. We need to instantiate it with the
        # right input types.
        output_types = [
            dtypes.DType(_.type) for _ in defined._signature.output_arg  # pylint: disable=protected-access
        ]
        # pylint: disable=protected-access
        defined._grad_func = self._grad_func.instantiate(input_types +
                                                         output_types)
        # pylint: enable=protected-access
      self._overload[key] = defined
    return defined

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Calls the overloaded function. Converts each positional argument to a tensor (raises a value
    # error if the result is not a tensor), collects their dtypes, instantiates (or reuses) the
    # specialization for those dtypes, and invokes it with the tensors and keyword arguments.
    # Returns what the specialized function returns. Adds ops and function definitions to the
    # default graph.
    input_types = []
    args = list(args)
    for (i, x) in enumerate(args):
      x = ops.convert_to_tensor(x)
      if not isinstance(x, tensor_lib.Tensor):
        raise ValueError(f"Expected a Tensor but got {x} with type {type(x)}.")
      input_types.append(x.dtype)
      args[i] = x
    return self.instantiate(input_types)(*args, **kwargs)


class _FuncGraph(ops.Graph):
  # [CLASS DOC]
  # HELPER: graph subclass used by the legacy Defun machinery to build a function body.
  # Tracks function inputs and outputs, intercepts op creation to capture tensors from other
  # graphs (as placeholders passed by the caller, or by copying ops), and hoists variables into the
  # outer graph through a custom getter. Instances are created by the tracing helper.
  """A helper for constructing a function.

  _FuncGraph overrides ops.Graph's create_op() so that we can keep
  track of all inputs into every op created inside the function.  If
  any input is from other graphs, we keep track of it in self.capture
  and substitute the input with a place holder.

  Each captured input's corresponding place holder is converted into a
  function argument and the caller passes in the captured tensor.
  """

  def __init__(self, name, capture_by_value, allowlisted_stateful_ops,
               capture_resource_var_by_value, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: name of the function; capture_by_value (copy captured values into the
    # body); allowlisted_stateful_ops (stateful ops permitted when capturing by value);
    # capture_resource_var_by_value (read resource variables outside and pass the value); extra
    # positional and keyword arguments go to the base graph constructor. Records the current
    # default graph as the outer graph, the current variable scope and its custom getter, and
    # initializes empty lists for inputs, outputs, extra inputs, extra arguments and extra
    # variables plus an empty capture table.
    super(_FuncGraph, self).__init__(*args, **kwargs)
    self._capture_by_value = capture_by_value
    self._allowlisted_stateful_ops = allowlisted_stateful_ops
    self._capture_resource_var_by_value = capture_resource_var_by_value
    self._building_function = True
    self._outer_graph = ops.get_default_graph()
    self._vscope = vs.get_variable_scope()
    self._old_custom_getter = self._vscope.custom_getter

    # The name of the function.
    self.name = name
    # Placeholder tensors representing the inputs to this function. The tensors
    # are in this _FuncGraph.
    self.inputs = []
    # Tensors that will be returned this function. The tensors are in this
    # _FuncGraph.
    self.outputs = []
    # Maps external tensor -> internal tensor (e.g. input placeholder).
    self._captured = {}
    # The external tensors that have been captured as inputs and must be passed
    # to this function (empty if capturing by value, otherwise these are the
    # keys of _captured).
    self.extra_inputs = []
    # Input placeholders that been added for captured values (empty if capturing
    # by value).
    self.extra_args = []
    # Captured variables.
    # TODO(skyewm): is this needed?
    self.extra_vars = []

  # pylint: disable=g-doc-return-or-yield

  @property
  def outer_graph(self):
    # [FUNCTION DOC]
    # Property returning the graph that was the default when this function graph was created.
    """The graph active when this _FuncGraph was created."""
    return self._outer_graph

  @tf_contextlib.contextmanager
  def container(self, container_name):
    # [FUNCTION DOC]
    # Context manager (generator) selecting the resource container. Input: container_name string.
    # Saves the current container of this graph and of the init-scope graph, sets both to the new
    # name, yields the new name, and restores both originals on exit even on error.
    """Returns a context manager that specifies the resource container to use.

    Overridden from `tf.Graph` to update both the init_scope container
    and the present inner container. This is necessary to make sure setting
    containers applies correctly both to created variables and to stateful
    ops.

    Args:
      container_name: container name string.

    Returns:
      A context manager for defining resource containers for stateful ops,
        yields the container name.
    """
    original_container = self._container
    # pylint: disable=protected-access
    with ops.init_scope():
      original_init_container = ops.get_default_graph()._container
    try:
      self._container = container_name
      with ops.init_scope():
        ops.get_default_graph()._container = container_name
      yield self._container
    finally:
      self._container = original_container
      with ops.init_scope():
        ops.get_default_graph()._container = original_init_container
    # pylint: enable=protected-access

  # pylint: enable=g-doc-return-or-yield

  def getvar(
      self,
      getter,
      name,
      shape=None,
      dtype=None,
      initializer=None,
      reuse=None,
      trainable=True,
      collections=None,  # pylint: disable=redefined-outer-name
      use_resource=None,
      **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the given getter argument and extra keyword arguments are ignored; the variable
    # is always fetched through the stored outer variable scope.
    # Custom variable getter installed while tracing a function body. Switches to the outer graph,
    # asks the saved variable scope for the variable (using the given name, shape, dtype,
    # initializer, reuse, trainable, collections and use_resource), records it in the extra
    # variables list, and returns either the variable's current value (for resource variables when
    # capture-by-value of resource variables is on) or the variable itself.
    """A custom variable getter."""
    # Here, we switch the default graph to the outer graph and ask the
    # variable scope in which the function is defined to give us the
    # variable. The variable is stashed in extra_vars and returned to
    # the caller.
    #
    # We capture these variables so that the variable definition is
    # hoisted upward to the outer most graph.
    with self._outer_graph.as_default():
      # pylint: disable=protected-access
      var = self._vscope.get_variable(
          vs._get_default_variable_store(),
          name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          use_resource=use_resource)
      self.extra_vars.append(var)
      if (isinstance(var, resource_variable_ops.BaseResourceVariable) and
          self._capture_resource_var_by_value):
        # For resource-based variables read the variable outside the function
        # and pass in the value. This ensures that the function is pure and
        # differentiable. TODO(apassos) this may have performance problems if
        # the function will only do embedding lookups on the variable.
        return var.value()
      return var

  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # [FUNCTION DOC]
    # Overrides op creation so every input that is an eager tensor or belongs to another graph is
    # first captured (replaced by a placeholder or copy inside this graph). Parameters are the same
    # as the base graph's internal creation method (op type, inputs list, optional dtypes, input
    # types, name, attributes, op definition, compute-device flag). Mutates the inputs list in
    # place and returns the created operation.
    for i, x in enumerate(inputs):
      if isinstance(x, ops.EagerTensor) or x.graph is not self:
        inputs[i] = self.capture(x)
    return super(_FuncGraph, self)._create_op_internal(
        op_type,
        inputs,
        dtypes=dtypes,
        input_types=input_types,
        name=name,
        attrs=attrs,
        op_def=op_def,
        compute_device=compute_device)

  def capture(self, tensor, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name argument is only honored when capturing by reference; with
    # capture-by-value it is ignored.
    # Makes an outside tensor usable inside this graph. Inputs: tensor to capture; name optional
    # placeholder name. Returns the already-captured counterpart if present; otherwise, when
    # capturing by value, copies the producing ops into the graph; else creates an extra-input
    # placeholder. Side effects: updates the capture table and related lists.
    """Adds the given tensor to this graph and returns the captured tensor."""
    if tensor.ref() in self._captured:
      # Captured already.
      return self._captured[tensor.ref()]
    elif self._capture_by_value:
      return self._add_tensor_and_parents(tensor)
    else:
      return self._capture_tensor_as_extra_input(tensor, name)

  @property
  def captures(self):
    # [FUNCTION DOC]
    # Property returning a list of pairs (outside tensor, inside placeholder or copy) for every
    # captured tensor, rebuilt from the capture table on each access.
    """Pairs of tensors and captured tensor."""
    return [(k.deref(), v) for k, v in self._captured.items()]

  def _capture_tensor_as_extra_input(self, tensor, name=None):
    # [FUNCTION DOC]
    # HELPER: capture-by-reference glue. Replaces an outside tensor with a new placeholder that
    # becomes an extra function argument. Inputs: tensor is the outside tensor (eager or graph);
    # name is an optional placeholder name. Records the tensor in the extra inputs, creates the
    # placeholder outside any control-flow context, copies resource handle shape/type data from the
    # tensor to the placeholder, and records it in inputs, the capture table and extra arguments.
    # Returns the placeholder, wrapped in a guarantee-constant op if the tensor is known constant.
    # Substitute with a placeholder.
    self.extra_inputs.append(tensor)
    # Hoist the new input placeholder out of any control flow context
    # we're currently in.
    with ops.control_dependencies(None):
      ph = array_ops.placeholder(
          tensor.dtype, shape=tensor.get_shape(), name=name)
    # pylint: disable=protected-access
    if isinstance(tensor, ops.EagerTensor):
      handle_data = tensor._handle_data
      if handle_data:
        handle_data = handle_data.SerializeToString()
    else:
      with tensor.graph._c_graph.get() as c_graph:
        handle_data = c_api.GetHandleShapeAndType(c_graph,
                                                  tensor._as_tf_output())

    if handle_data:
      with ph.graph._c_graph.get() as c_graph:
        c_api.SetHandleShapeAndType(c_graph, ph._as_tf_output(),
                                    compat.as_bytes(handle_data))
    # pylint: enable=protected-access
    self.inputs.append(ph)
    self._captured[tensor.ref()] = ph
    self.extra_args.append(ph)
    if _is_guaranteed_const(tensor):
      with ops.control_dependencies(None):
        return array_ops.guarantee_const(ph)
    else:
      return ph

  def _add_tensor_and_parents(self, tensor):
    # [FUNCTION DOC]
    # HELPER: capture-by-value glue. Copies the operation producing the given outside tensor (and
    # its ancestors) into this graph and returns the matching output tensor of the copy.
    op = self._add_op_and_parents(tensor.op)
    return op.outputs[tensor.value_index]

  def _add_op_and_parents(self, op: ops.Operation):
    # [FUNCTION DOC]
    # HELPER: recursive step of capture-by-value.
    # KNOWN ISSUE: the stateful check tests membership of the operation itself in the allowlist,
    # while the allowlist is described as a set of ops; callers passing names would never match.
    # Copies an outside operation and all of its producers into this graph. Input: op is the
    # operation. Raises a value error if it is stateful and not allowlisted, or if it is a
    # placeholder. Recursively copies inputs, creates the copy with same type, output dtypes, name
    # and attributes, records the output mapping in the capture table, and returns the new op.
    # pylint: disable=protected-access
    op_def = graph_to_function_def._get_op_def(op)
    if op._is_stateful and op not in self._allowlisted_stateful_ops:
      raise ValueError(f"Cannot capture a stateful node (name:{op.name}, "
                       f"type:{op.type}) by value.")
    elif op.type in ("Placeholder", "PlaceholderV2"):
      raise ValueError(f"Cannot capture a placeholder (name:{op.name}, "
                       f"type:{op.type}) by value.")
    # pylint: enable=protected-access

    captured_inputs = [self._add_tensor_and_parents(x) for x in op.inputs]

    captured_op = self._create_op_internal(
        op.type,
        captured_inputs, [o.dtype for o in op.outputs],
        name=op.name,
        attrs=op.node_def.attr,
        op_def=op_def)

    for t, captured_t in zip(op.outputs, captured_op.outputs):
      self._captured[t.ref()] = captured_t

    return captured_op


def func_graph_from_py_func(func,
                            arg_names,
                            arg_types,
                            name=None,
                            capture_by_value=False,
                            device=None,
                            colocation_stack=None,
                            container=None,
                            collections_ref=None,
                            arg_shapes=None,
                            allowlisted_stateful_ops=None,
                            capture_resource_var_by_value=True):
  # [FUNCTION DOC]
  # HELPER: internal builder that traces a python callable into a function-building graph.
  # Inputs: func is the callable; arg_names, arg_types and optional arg_shapes (default all
  # unknown) describe placeholders created for each argument; name defaults to the callable's
  # name; capture_by_value copies captured values into the body; device is a device name or
  # function applied while building; colocation_stack, container and collections_ref optionally
  # preset those graph properties; allowlisted_stateful_ops and capture_resource_var_by_value
  # configure capture behavior.
  # Behavior: makes the graph default, creates placeholders, runs the callable under a variable
  # scope whose custom getter hoists variables to the outer graph, treats a single null result as
  # no outputs, wraps a single result into a tuple, converts results to tensors and captures
  # results that belong to other graphs. Returns the function graph with inputs and outputs set.
  # Raises a value error if any output is null. Side effects: runs user code that adds ops and
  # variables; may mutate the supplied collections reference.
  """Returns a _FuncGraph generated from `func`.

  Args:
    func: A Python callable which constructs a TF function body. The arguments
      must correspond to `arg_types`. Returns a value or list/tuple of values.
      No returned value can be None.
    arg_names: A sequence of strings for the function argument names.
    arg_types: A sequence of the function's argument types.
    name: The function name. If None, the name is derived from `func`.
    capture_by_value: boolean. If True, captured values will be copied into the
      function body.
    device: device name or function.
    colocation_stack: A colocation stack (list) the _FuncGraph should use.
    container: A container name the _FuncGraph should start with.
    collections_ref: A reference to a collections dict the _FuncGraph should
      use internally.
    arg_shapes: A sequence of the function's argument shapes.
    allowlisted_stateful_ops: A set of ops that if stateful we ignore and
      re-create.
    capture_resource_var_by_value: Boolean (defaults to True). If False,
      captured resource variable returns the handle instead of value.

  Returns:
    A _FuncGraph.

  Raises:
    ValueError: if func returns None.
  """
  if not name:
    name = function_utils.get_func_name(func)
  func_graph = _FuncGraph(name, capture_by_value, allowlisted_stateful_ops,
                          capture_resource_var_by_value)

  with func_graph.as_default(), ops.device(device):
    # pylint: disable=protected-access
    if collections_ref is not None:
      func_graph._collections = collections_ref
    if container is not None:
      func_graph._container = container
    if colocation_stack is not None:
      func_graph._colocation_stack = colocation_stack
    # pylint: enable=protected-access

    if arg_shapes is None:
      arg_shapes = [None] * len(arg_types)

    # Create placeholders for the function arguments.
    for (argname, argtype, argshape) in zip(arg_names, arg_types, arg_shapes):
      argholder = array_ops.placeholder(argtype, shape=argshape, name=argname)
      func_graph.inputs.append(argholder)
    # Call func and gather the output tensors.
    with vs.variable_scope("", custom_getter=func_graph.getvar):
      outputs = func(*func_graph.inputs)

    # There is no way of distinguishing between a function not returning
    # anything and a function returning None in Python.
    # We need to allow the former and ideally want to forbid the latter as
    # it is most likely user error.
    # TODO(iga): Consider adding a @NoOutput decorator on top of @Defun to
    # allow users to explicitly mark the function as not returning anything.
    # For now, we allow a single None return and interpret it as a function
    # with no output.
    if outputs is None:
      outputs = []
    else:
      # If func only returned one value, make it a tuple.
      if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
      if any(_ is None for _ in outputs):
        raise ValueError(f"Function {name} can not return None.")
    # Ensures each output is a Tensor in the function graph.
    outputs = [ops.convert_to_tensor(t) for t in outputs]
    outputs = [func_graph.capture(t) if t.graph is not func_graph else t
               for t in outputs]
    func_graph.outputs = outputs
  return func_graph


def _is_guaranteed_const(tensor):
  # [FUNCTION DOC]
  # HELPER: internal check used when capturing external tensors as function inputs.
  # KNOWN ISSUE: the docstring says "children" but the code walks the producing ops of inputs
  # (ancestors). Eager tensors are never considered constant.
  # Input: tensor is a graph tensor. Returns true if its producing op is a guarantee-constant op,
  # or if the op has inputs and all of them are (recursively) guaranteed constant; false otherwise.
  # Uses an explicit stack with post-order revisits, so it does not recurse. No side effects.
  """Determines whether `tensor` is guaranteed to be a constant.

  A tensor is guaranteed to be a constant if either it was produced by
  a `GuaranteeConst` op or if all of its children are guaranteed to be
  constants.

  Args:
    tensor: The tensor for which to determine const-ness.

  Returns:
    True if `tensor` is guaranteed to be a constant, False otherwise.
  """

  if isinstance(tensor, ops.EagerTensor):
    return False

  class Work(object):
    # [CLASS DOC]
    # HELPER: work item for the iterative depth-first walk in the constant check; holds an
    # operation and a flag for the post-order revisit.

    def __init__(self, op: ops.Operation, leaving):
      # [FUNCTION DOC]
      # Stores the operation to visit and a flag saying whether this is the second (post-order)
      # visit, after all of its inputs were examined.
      self.op = op
      self.leaving = leaving

  is_guaranteed_const = lambda op: op.node_def.op == "GuaranteeConst"
  constants = set([])
  def all_inputs_const(op: ops.Operation):
    # [FUNCTION DOC]
    # HELPER: nested predicate for the constant check. Returns true (a truthy value) if the
    # operation has at least one input and every input's producing op is already known constant;
    # ops with no inputs yield a falsy value. Reads the enclosing set of constant ops.
    # If all inputs of an op are guaranteed constants, then we can infer that
    # the op produces a constant as well.
    return op.inputs and all(inp.op in constants for inp in op.inputs)

  visited = set([])
  stack = [Work(tensor.op, leaving=False)]
  while stack:
    work = stack.pop()
    if work.leaving:
      if all_inputs_const(work.op):
        constants.add(work.op)
      continue
    visited.add(work.op)
    if is_guaranteed_const(work.op):
      constants.add(work.op)
      continue

    # This op will be revisited after all its inputs are checked for const-ness.
    stack.append(Work(work.op, leaving=True))
    for inp in work.op.inputs:
      if inp.op not in visited:
        stack.append(Work(inp.op, leaving=False))
  return tensor.op in constants


def _call(sig, *inputs, **kwargs):
  # [FUNCTION DOC]
  # HELPER: internal glue that emits a function-call op into the default graph.
  # KNOWN ISSUE: the docstring text about the signature type ("OpDefArg.a ...") is garbled; the
  # argument is really an op definition (signature) proto, not a defined-function object.
  # Inputs: sig is the function signature; inputs are tensors passed as arguments (count must match
  # the signature); keyword options may include name (defaults to the function name) and others
  # handled as attributes (such as noinline). Creates the call op with output types from the
  # signature. Returns a pair: the result (single tensor, tuple of tensors, or the operation when
  # there are no outputs) and the operation. Raises a value error on argument count mismatch or
  # unknown keywords. Side effect: adds an operation to the default graph.
  """Adds a node calling a function.

  This adds a `call` op to the default graph that calls the function
  of signature `sig`, passing the tensors in `inputs` as arguments.
  It returns the outputs of the call, which are one or more tensors.

  `sig` is OpDefArg.a `_DefinedFunction` object.

  You can pass an optional keyword parameter `name=string` to name the
  added operation.

  You can pass an optional keyword parameter `noinline=True|False` to
  instruct the runtime not to inline the function body into the call
  site.

  Args:
    sig: OpDefArg. The signature of the function.
    *inputs: arguments to the function.
    **kwargs: Optional keyword arguments.  Can only contain 'name' or
        'noinline'.

  Returns:
     A 2-element tuple. First element: a Tensor if the function returns a single
     value; a list of Tensors if the function returns multiple value; the
     Operation if the function returns no values. Second element: the Operation.

  Raises:
    ValueError: if the arguments are invalid.
  """
  if len(inputs) != len(sig.input_arg):
    raise ValueError(f"Expected {len(sig.input_arg):d} arguments, got "
                     f"{len(inputs):d}.")
  name = kwargs.pop("name", None)
  g = ops.get_default_graph()
  func_name = sig.name
  if name is None:
    name = func_name
  attrs = _parse_kwargs_as_attrs(func_name, **kwargs)
  output_types = [dtypes.DType(x.type) for x in sig.output_arg]
  op = g._create_op_internal(  # pylint: disable=protected-access
      func_name, list(inputs), output_types, name=name, attrs=attrs, op_def=sig)
  if op.outputs:
    if len(op.outputs) == 1:
      ret = op.outputs[0]
    else:
      ret = tuple(op.outputs)
  else:
    ret = op
  return ret, op


def _from_definition(fdef, grad_func=None):
  # [FUNCTION DOC]
  # HELPER: internal factory used by the library loader.
  # Creates a defined-function object from an existing function definition proto, without any
  # python callable. Inputs: fdef is the function definition; grad_func is an optional defined
  # function to use as its gradient. Argument names, types, function name and output names come
  # from the signature; python gradient functions are not recoverable and stay unset.
  # Imports the definition into a C-level function (serialized or not depending on the build
  # flag), and sets the captured inputs empty and the cached op definition to the signature.
  # Returns the new defined-function object.
  """Creates a _DefinedFunction initialized from a FunctionDef proto.

  Args:
    fdef: a FunctionDef
    grad_func: a _DefinedFunction or None

  Returns:
    A _DefinedFunction representing fdef
  """
  # TODO(iga): This method does major surgery on _DefinedFunction.
  # Make it a named constructor using @classmethod of _DefinedFunction.

  # The Python callable is only needed to create a FunctionDef. Since we have
  # the FunctionDef here, we don't need to set _DefinedFunction._func (nor do we
  # have access to such a callable here).
  func = None
  argnames = [arg.name for arg in fdef.signature.input_arg]
  input_types = tuple(
      dtypes.as_dtype(arg.type) for arg in fdef.signature.input_arg)
  func_name = fdef.signature.name
  # Note: FunctionDefs do not include python gradient functions, so if the
  # original _DefinedFunction included one it will not be reflected here.
  python_grad_func = None
  out_names = [arg.name for arg in fdef.signature.output_arg]
  result = _DefinedFunction(func, argnames, input_types, func_name, grad_func,
                            python_grad_func, out_names)
  # pylint: disable=protected-access
  if is_oss:
    serialized = fdef.SerializeToString()
    c_func = c_api.TF_FunctionImportFunctionDef(serialized)
  else:
    c_func = c_api.TF_FunctionImportFunctionDefNoSerialization(fdef)
  result._c_func = c_api_util.ScopedTFFunction(c_func, func_name)
  result._extra_inputs = []
  result._op_def = fdef.signature
  # pylint: enable=protected-access

  return result


def from_library(lib):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a list is returned, but a non-empty library yields a dict
  # values view (only an empty library yields a list). Also, a library whose functions all have
  # gradients raises the cyclic error, and functions in unreachable cycles are silently dropped.
  # Builds defined-function objects from every function in a function library proto, wiring each
  # function to its gradient function. Input: lib is a function definition library proto.
  # Validates that every gradient entry names functions present in the library (raises a value
  # error otherwise), then instantiates functions without gradients first and, repeatedly, those
  # whose gradient function is already built. Raises a value error if no function lacks a
  # gradient (cyclic). Returns an empty list for an empty library, else the created functions.
  # Side effects: each import registers a C-level function object for each definition.
  """Creates _DefinedFunctions initialized from a FunctionDefLibrary proto.

  This method handles assigning the correct gradient functions to each
  function.

  Args:
    lib: a FunctionDefLibrary

  Returns:
    A list of _DefinedFunctions

  Raises:
    ValueError: `lib` is invalid
  """
  if not lib.function and not lib.gradient:
    return []

  # function name -> FunctionDef proto
  funcs = {fdef.signature.name: fdef for fdef in lib.function}

  # Validate that all references function names have function defs
  for g in lib.gradient:
    if g.function_name not in funcs:
      raise ValueError(f"FunctionDefLibrary missing '{g.function_name}' "
                       f"FunctionDef\n{lib}")
    if g.gradient_func not in funcs:
      raise ValueError(f"FunctionDefLibrary missing '{g.gradient_func}' "
                       f"FunctionDef\n{lib}")

  # function name -> gradient function name
  func_to_grad = collections.defaultdict(lambda: None)
  # gradient function name -> names of functions having that grad function
  grad_to_funcs = collections.defaultdict(list)

  for gdef in lib.gradient:
    func_to_grad[gdef.function_name] = gdef.gradient_func
    grad_to_funcs[gdef.gradient_func].append(gdef.function_name)

  # Start with functions without gradients
  ready = [
      fdef for fdef in lib.function if func_to_grad[fdef.signature.name] is None
  ]
  if not ready:
    raise ValueError(
        f"FunctionDefLibrary contains cyclic gradient functions!\n{lib}")
  # function name -> _DefinedFunction
  initialized = {}

  while ready:
    fdef = ready.pop()
    name = fdef.signature.name

    grad = initialized.get(func_to_grad[name])
    if func_to_grad[name]:
      assert grad
    defined_func = _from_definition(fdef, grad_func=grad)
    initialized[name] = defined_func

    ready.extend(funcs[f] for f in grad_to_funcs[name])

  return initialized.values()


def _get_experimental_kwarg_as_attr(attr_name, value):
  # [FUNCTION DOC]
  # HELPER: internal converter for experimental keyword options.
  # KNOWN ISSUE: the docstring says it handles any python object, but only boolean, integer, float
  # and string values are accepted. Booleans are checked first so they are not treated as integers.
  # Inputs: attr_name is the option name (used in the message); value is the option value.
  # Returns the matching attribute value proto (strings stored as bytes).
  # Raises a value error for any other value type.
  """Creates an AttrValue for a python object."""
  if isinstance(value, bool):
    return attr_value_pb2.AttrValue(b=value)
  elif isinstance(value, int):
    return attr_value_pb2.AttrValue(i=value)
  elif isinstance(value, float):
    return attr_value_pb2.AttrValue(f=value)
  elif isinstance(value, str):
    return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
  else:
    raise ValueError(f"Attribute {attr_name} must be bool, int, float, or "
                     f"str. Got {type(value)}.")


def _get_kwarg_as_str_attr(attr_name, value):
  # [FUNCTION DOC]
  # HELPER: internal converter for string-only keyword options.
  # KNOWN ISSUE: the docstring says it creates a value for any python object, but only strings are
  # accepted.
  # Inputs: attr_name is the option name (used in the message); value must be a string.
  # Returns an attribute value proto holding the string as bytes. Raises a value error otherwise.
  """Creates an AttrValue for a python object."""
  if isinstance(value, str):
    return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
  else:
    raise ValueError(f"Attribute {attr_name} must be str. Got {type(value)}.")


def _parse_kwargs_as_attrs(func_name, **kwargs):
  # [FUNCTION DOC]
  # HELPER: internal glue converting decorator keyword options into attribute values for function
  # definitions and call ops.
  # Inputs: func_name is the function name (used to build an XLA scope name); keyword options may
  # include noinline (boolean), compiled and separate_compiled_gradients (booleans), options with
  # the experimental prefix (bool, int, float or string), and the implements and reference string
  # options. Always adds an attribute disabling call shape inference. When compiled is given, adds
  # XLA compile flags and an XLA scope (inherited from the default graph's attribute scope if set,
  # else derived from the function name).
  # Returns a mapping from attribute names to attribute value protos.
  # Raises a value error for unsupported value types or any unrecognized keyword.
  # Reads the default graph's private attribute scope map; does not mutate the caller's mapping.
  """Parses **kwargs into a node's attributes."""
  attrs = {}

  noinline = kwargs.pop("noinline", None)
  if noinline is not None:
    attrs["_noinline"] = attr_value_pb2.AttrValue(b=bool(noinline))

  # For compatibility with previous behavior, Defun does not perform shape
  # inference through its function call operations.
  attrs["_disable_call_shape_inference"] = attr_value_pb2.AttrValue(b=True)

  compiled = kwargs.pop("compiled", None)
  separate_compiled_gradients = kwargs.pop("separate_compiled_gradients", None)
  if compiled is not None:
    attrs["_XlaCompile"] = attr_value_pb2.AttrValue(b=bool(compiled))
    attrs["_XlaSeparateCompiledGradients"] = attr_value_pb2.AttrValue(
        b=bool(separate_compiled_gradients))
    # Forward _XlaScope from enclosing context (if set), otherwise create new.
    # pylint: disable=protected-access
    if "_XlaScope" in ops.get_default_graph()._attr_scope_map:
      attrs["_XlaScope"] = ops.get_default_graph()._attr_scope_map["_XlaScope"]
    else:
      attrs["_XlaScope"] = attr_value_pb2.AttrValue(
          s=("function_%s" % func_name).encode())
    # pylint: enable=protected-access

  kwargs_keys = list(kwargs.keys())
  for key in kwargs_keys:
    if key.startswith("experimental_"):
      attrs[key] = _get_experimental_kwarg_as_attr(key, kwargs[key])
      del kwargs[key]
    # Support for https://github.com/tensorflow/community/pull/113/files.
    elif key == "_implements" or key == "_reference":
      attrs[key] = _get_kwarg_as_str_attr(key, kwargs[key])
      del kwargs[key]
  if kwargs:
    raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}.")
  return attrs


def get_extra_vars():
  # [FUNCTION DOC]
  # Returns the variables created inside the function body currently being defined (hoisted to the
  # outer graph). Takes nothing. Reads the default graph; if it is a function-building graph
  # returns its live list of captured variables, otherwise returns an empty list.
  """Returns the captured variables by the function.

  Returns:
    If the default graph is being used to define a function, the
    returned list of variables are those created inside the function
    body so far. Otherwise, returns an empty list.
  """
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_vars
  else:
    return []


def get_extra_inputs():
  # [FUNCTION DOC]
  # Returns the outside tensors captured so far by the function body currently being defined.
  # Takes nothing. Reads the default graph; if it is a function-building graph returns its live
  # list of captured external tensors, otherwise returns an empty list. No side effects.
  """Returns the captured input tensors by the function.

  Returns:
    If the default graph is being used to define a function, the
    returned list of tensors are those accessed inside the function body
    but defined outside the function body so far. Otherwise, returns an
    empty list.
  """
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_inputs
  else:
    return []


def get_extra_args():
  # [FUNCTION DOC]
  # Returns the placeholder tensors, inside the function body being defined, that stand for
  # captured outside tensors (parallel to the list from the extra-inputs getter). Takes nothing.
  # Reads the current default graph; if it is a function-building graph returns its list of
  # extra argument placeholders (the live list, not a copy), otherwise returns an empty list.
  """Returns the corresponding function arguments for the captured inputs.

  Returns:
    If the default graph is being used to define a function, the
    returned list of place holders are those used inside the function
    body corresponding those returned by get_extra_inputs(). Otherwise,
    returns an empty list.
  """
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_args
  else:
    return []


def _type_list_to_str(types):
  # [FUNCTION DOC]
  # HELPER: internal utility building cache keys for overloaded functions.
  # KNOWN ISSUE: the dtype-to-string table maps both unsigned 8-bit and signed 8-bit integers to
  # the same code, so overload keys for those two types collide and an instantiation for one may
  # wrongly be reused for the other.
  # Converts a list of data types into one concatenated short-code string.
  # Input: types is a sequence of data types. Returns the concatenation of per-type codes.
  # Raises a value error listing the unsupported types if any type is not in the table.
  if any(_ not in _DTYPE_TO_STR for _ in types):
    unsupported_types = [type_ for type_ in types if type_ not in _DTYPE_TO_STR]
    raise ValueError(f"Unsupported dtypes {unsupported_types} in "
                     "`types`. Supported dtypes are "
                     f"{_DTYPE_TO_STR.keys()}.")
  return "".join(_DTYPE_TO_STR[_] for _ in types)


# NOTE: The list needs to be extended when more data types are added.
_DTYPE_TO_STR = {
    dtypes.float16: "f16",
    dtypes.float32: "f32",
    dtypes.float64: "f64",
    dtypes.int32: "i32",
    dtypes.uint8: "i8",
    dtypes.uint16: "u16",
    dtypes.uint32: "u32",
    dtypes.uint64: "u64",
    dtypes.int16: "i16",
    dtypes.int8: "i8",
    dtypes.string: "s",
    dtypes.complex64: "c64",
    dtypes.complex128: "c128",
    dtypes.int64: "i64",
    dtypes.bool: "b",
    dtypes.qint8: "qi8",
    dtypes.quint8: "qu8",
    dtypes.qint16: "qi16",
    dtypes.quint16: "qu16",
    dtypes.qint32: "qi32",
    dtypes.bfloat16: "b16",
    dtypes.float8_e5m2: "f8e5m2",
    dtypes.float8_e4m3fn: "f8e4m3fn",
    dtypes.float8_e4m3fnuz: "f8e4m3fnuz",
    dtypes.float8_e4m3b11fnuz: "f8e4m3b11fnuz",
    dtypes.float8_e5m2fnuz: "f8e5m2fnuz",
    dtypes.float4_e2m1fn: "f4e2m1fn",
    dtypes.int4: "i4",
    dtypes.uint4: "u4",
    dtypes.int2: "i2",
    dtypes.uint2: "u2",
}
