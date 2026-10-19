# [MODULE DOC]
# Defines the legacy tf.layers base Layer class, built on top of the Keras base layer. It adds
# variable-scope based variable management (reuse, scope naming, graph collections for updates
# and regularization losses) on top of Keras, plus a global "Keras style" switch.
# Exposes keras_style_scope (context manager), set_keras_style (permanent global switch),
# the Layer class, and re-exports InputSpec for backward compatibility. Private helpers
# check the style flag and add elements to graph collections.
# Import-time side effects: only defines a module-level boolean flag (initially off).
# Not a helper module; it is the base for the other legacy layer modules in this package.
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
# =============================================================================
# pylint: disable=g-classes-have-attributes
"""Contains the base Layer class, from which all layers inherit."""
import copy
import warnings

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.legacy_tf_layers import variable_scope_shim
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest

# Avoid breaking users who directly import this symbol from this file.
# TODO(fchollet): remove this.
InputSpec = base_layer.InputSpec  # pylint: disable=invalid-name

_KERAS_STYLE_SCOPE = False


@tf_contextlib.contextmanager
def keras_style_scope():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it "Yields: a keras layer style scope", but the generator
  # yields nothing (no value is bound by the with statement).
  # Context manager that temporarily turns on the module-wide Keras-style flag. Takes no
  # inputs and yields nothing. On entry it remembers the previous flag value and sets the flag
  # on; on exit (even on error) it restores the previous value. Layers created inside use
  # Keras-style variable management; passing a scope or reuse to them then raises an error.
  # Side effect: mutates the module-level flag.
  """Use Keras-style variable management.

  All tf.layers and tf RNN cells created in this scope use Keras-style
  variable management.  Creating such layers with a scope= argument is
  disallowed, and reuse=True is disallowed.

  The purpose of this scope is to allow users of existing layers to
  slowly transition to a Keras layers API without breaking existing
  functionality.

  One example of this is when using TensorFlow's RNN classes with Keras
  Models or Networks.  Because Keras models do not properly set variable
  scopes, users of RNNs may either accidentally share scopes between two
  different models, or get errors about variables that already exist.

  Example:

  ```python
  class RNNModel(tf.keras.Model):

    def __init__(self, name):
      super(RNNModel, self).__init__(name=name)
      self.rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
        [tf.compat.v1.nn.rnn_cell.LSTMCell(64) for _ in range(2)])

    def call(self, input, state):
      return self.rnn(input, state)

  model_1 = RNNModel("model_1")
  model_2 = RNNModel("model_2")

  # OK
  output_1, next_state_1 = model_1(input, state)
  # Raises an error about trying to create an already existing variable.
  output_2, next_state_2 = model_2(input, state)
  ```

  The solution is to wrap the model construction and execution in a keras-style
  scope:

  ```python
  with keras_style_scope():
    model_1 = RNNModel("model_1")
    model_2 = RNNModel("model_2")

    # model_1 and model_2 are guaranteed to create their own variables.
    output_1, next_state_1 = model_1(input, state)
    output_2, next_state_2 = model_2(input, state)

    assert len(model_1.weights) > 0
    assert len(model_2.weights) > 0
    assert(model_1.weights != model_2.weights)
  ```

  Yields:
    A keras layer style scope.
  """
  global _KERAS_STYLE_SCOPE
  stack = _KERAS_STYLE_SCOPE
  _KERAS_STYLE_SCOPE = True
  try:
    yield
  finally:
    _KERAS_STYLE_SCOPE = stack


def set_keras_style():
  # [FUNCTION DOC]
  # Permanently switches the module-wide Keras-style flag on for the rest of the program.
  # Takes no inputs, returns nothing, raises nothing. Layers constructed afterwards use
  # Keras-style variable management. There is no function to turn it off again.
  """Use Keras-style variable management.

  All tf.layers and tf RNN cells created after keras style ha been enabled
  use Keras-style variable management.  Creating such layers with a
  scope= argument is disallowed, and reuse=True is disallowed.

  The purpose of this function is to allow users of existing layers to
  slowly transition to Keras layers API without breaking existing
  functionality.

  For more details, see the documentation for `keras_style_scope`.

  Note, once keras style has been set, it is set globally for the entire
  program and cannot be unset.

  Example:

  ```python
  set_keras_style()

  model_1 = RNNModel(name="model_1")
  model_2 = RNNModel(name="model_2")

  # model_1 and model_2 are guaranteed to create their own variables.
  output_1, next_state_1 = model_1(input, state)
  output_2, next_state_2 = model_2(input, state)

  assert len(model_1.weights) > 0
  assert len(model_2.weights) > 0
  assert(model_1.weights != model_2.weights)
  ```
  """
  global _KERAS_STYLE_SCOPE
  _KERAS_STYLE_SCOPE = True


def _is_in_keras_style_scope():
  # [FUNCTION DOC]
  # HELPER: used by the Layer constructor to decide between Keras-style and legacy variable
  # management.
  # Returns the current boolean value of the module-wide Keras-style flag. No inputs, no side
  # effects, no exceptions.
  global _KERAS_STYLE_SCOPE
  return _KERAS_STYLE_SCOPE


class Layer(base_layer.Layer):
  # [CLASS DOC]
  # Legacy base class for tf.layers layers; subclasses the Keras base Layer. It adds
  # variable-scope based variable creation and reuse (unless Keras style is active, in which
  # case it behaves like a Keras layer), lazily captured variable scopes, op naming from the
  # scope, graph collection updates (update ops, regularization losses) in graph mode, and a
  # custom deep copy. State: the captured scope, the reuse flag, the current scope, whether
  # the layer is Keras style (decided at construction), and whether call accepts a scope
  # argument. Instances are created by subclasses and used by calling them on inputs.
  """Base layer class.

  It is considered legacy, and we recommend the use of `tf.keras.layers.Layer`
  instead.

  Args:
    trainable: Boolean, whether the layer's variables should be trainable.
    name: String name of the layer.
    dtype: Default dtype of the layer's weights (default of `None` means use the
      type of the first input).

  Read-only properties:
    name: The name of the layer (string).
    dtype: Default dtype of the layer's weights (default of `None` means use the
      type of the first input).
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and
      non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.
    trainable_weights: List of variables to be included in backprop.
    non_trainable_weights: List of variables that should not be
      included in backprop.
    weights: The concatenation of the lists trainable_weights and
      non_trainable_weights (in this order).

  Mutable properties:
    trainable: Whether the layer should be trained (boolean).
    input_spec: Optional (list of) `InputSpec` object(s) specifying the
      constraints on inputs that can be accepted by the layer.
  """

  def __init__(self, trainable=True, name=None, dtype=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructs a legacy layer. Inputs: trainable (boolean, default true), name (optional
    # string), dtype (optional; when missing an infer-from-input policy is used so the dtype
    # follows the first input), and extra keyword arguments passed to the Keras base layer;
    # two private ones are consumed here: a pre-made variable scope and a reuse flag. Autocast
    # defaults to off. Disables Keras usage instrumentation, disables resource variables by
    # default, and if the Keras style flag is on marks the layer as Keras style.
    # Raises a value error if Keras style is on and a scope or a non-null reuse was given.
    # If a scope was given it is captured by entering a variable scope with it.
    # Side effects: sets several attributes on the instance; reads the global flag.
    # For backwards compatibility, legacy layers do not use `ResourceVariable`
    # by default.
    self._use_resource_variables = False
    scope = kwargs.pop('_scope', None)
    self._reuse = kwargs.pop('_reuse', None)

    # Avoid an incorrect lint error
    self._trainable_weights = []
    self.built = False

    if dtype is None:
      # Indicates to infer dtype from inputs. When the V2 dtype behavior is
      # enabled, Keras layers default their dtype to floatx instead, so we pass
      # an "_infer" policy to keep the old V1 behavior.
      dtype = policy.Policy('_infer')

    if 'autocast' not in kwargs:
      kwargs['autocast'] = False

    # Mark that legacy layers should not be instrumented as Keras usage
    self._disable_keras_instrumentation = True

    super(Layer, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                **kwargs)

    if _is_in_keras_style_scope():
      if scope is not None:
        raise ValueError(
            'scope argument not allowed when keras style layers are enabled, '
            'but saw: {}'.format(scope))
      if self._reuse is not None:
        raise ValueError(
            'reuse argument not allowed when keras style layers are enabled, '
            'but saw: {}'.format(self._reuse))
      self._keras_style = True
    else:
      self._keras_style = False

    self._call_has_scope_arg = 'scope' in self._call_fn_args
    if scope:
      with vs.variable_scope(scope) as captured_scope:
        self._scope = captured_scope
    else:
      self._scope = None
    self._current_scope = None

  # We no longer track graph in tf.layers layers. This property is only kept to
  # maintain API backward compatibility.
  @property
  def graph(self):
    # [FUNCTION DOC]
    # Deprecated read-only property kept for API compatibility. Always emits a warning that
    # it is deprecated. Raises a runtime error when executing eagerly; otherwise returns
    # nothing (null), because these layers no longer track a graph.
    warnings.warn('`Layer.graph` is deprecated and '
                  'will be removed in a future version. '
                  'Please stop using this property because tf.layers layers no '
                  'longer track their graph.')
    if context.executing_eagerly():
      raise RuntimeError('Layer.graph not supported when executing eagerly.')
    return None

  def _init_set_name(self, name):
    # [FUNCTION DOC]
    # HELPER: overrides the Keras base layer naming hook during construction.
    # Sets the layer name and base name. Input: name, which may be a string, a variable scope
    # object, or empty. For a variable scope, the base name is the scope's name and the layer
    # name is a freshly generated unique name. For a string, both use that string. If the name
    # is empty or null, a unique name derived from the class name is generated and used.
    # Side effects: sets the name and base name attributes; consumes a unique-name counter.
    # Determine layer name (non-unique).
    if isinstance(name, vs.VariableScope):
      base_name = name.name
      self._name, _ = self._make_unique_name()
    else:
      base_name = name
      self._name = name
    if not name:
      self._name, base_name = self._make_unique_name()
    self._base_name = base_name

  def _make_unique_name(self, name_uid_map=None, avoid_names=None,
                        namespace='', zero_based=False):
    # [FUNCTION DOC]
    # HELPER: name generation for _init_set_name.
    # Builds a unique layer name from the snake-case form of the class name using the Keras
    # backend unique-object-name facility. Inputs: optional mapping of name counters, optional
    # collection of names to avoid, a namespace string (default empty), and a zero-based flag
    # (default false) controlling whether numbering starts at zero. Returns a pair of the
    # unique name and the base (snake-case) name. Side effect: advances the name counter.
    base_name = base_layer.to_snake_case(self.__class__.__name__)
    name = backend.unique_object_name(
        base_name,
        name_uid_map=name_uid_map,
        avoid_names=avoid_names,
        namespace=namespace,
        zero_based=zero_based)
    return (name, base_name)

  @property
  def scope_name(self):
    # [FUNCTION DOC]
    # Read-only property returning the name of the layer's variable scope. Raises a value
    # error if the scope has not been captured yet, i.e. the layer has never been called (or
    # constructed with a scope), explaining that the layer must be called first.
    if not self._scope:
      raise ValueError('No name available for layer scope because the layer "' +
                       self._name + '" has not been used yet. The scope name ' +
                       ' is determined the first time the layer instance is ' +
                       'called. You must therefore call the layer before ' +
                       'querying `scope_name`.')
    return self._scope.name

  def add_loss(self, losses, inputs=None):
    # [FUNCTION DOC]
    # Adds loss tensors or callables to the layer by delegating to the Keras base class. Inputs:
    # losses (a tensor, callable, or list of them) and optional inputs the losses depend on.
    # In graph mode (not eager), newly added losses are additionally evaluated (callable ones
    # are called, non-null results kept) and appended to the global regularization-losses graph
    # collection. Returns nothing. Side effects: mutates the layer's loss lists and the graph
    # collection.
    previous_losses_length = len(self._losses)
    previous_callable_losses_length = len(self._callable_losses)
    super(Layer, self).add_loss(losses, inputs=inputs)
    if not context.executing_eagerly():
      # TODO(fchollet): deprecate collection below.
      new_losses = self._losses[previous_losses_length:]
      new_callable_losses = self._callable_losses[
          previous_callable_losses_length:]
      for regularizer in new_callable_losses:
        loss_tensor = regularizer()
        if loss_tensor is not None:
          new_losses.append(loss_tensor)
      _add_elements_to_collection(
          new_losses,
          ops.GraphKeys.REGULARIZATION_LOSSES)

  def _name_scope(self):  # pylint: disable=method-hidden
    # [FUNCTION DOC]
    # HELPER: overrides the Keras name-scope hook. Returns the name scope string to use for
    # ops: the Keras default when the layer is Keras style, otherwise the original name scope
    # of the current variable scope (which must already be set, else an attribute error on a
    # null value would occur).
    """Determines op naming for the Layer."""
    if self._keras_style:
      return super(Layer, self)._name_scope()
    return self._current_scope.original_name_scope

  def _set_scope(self, scope=None):
    # [FUNCTION DOC]
    # HELPER: lazily captures the layer's variable scope on first use. Input: optional scope
    # (a name or scope object). Does nothing if a scope is already set. Otherwise, if reuse is
    # truthy, enters a variable scope using the given scope or, when null, the base name;
    # if not reuse, enters a variable scope with the base name as default name (making it
    # unique). The captured scope is stored on the layer. Returns nothing.
    if self._scope is None:
      # If constructed with _scope=None, lazy setting of scope.
      if self._reuse:
        with vs.variable_scope(
            scope if scope is not None else self._base_name) as captured_scope:
          self._scope = captured_scope
      else:
        with vs.variable_scope(
            scope, default_name=self._base_name) as captured_scope:
          self._scope = captured_scope

  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 use_resource=None,
                 synchronization=vs.VariableSynchronization.AUTO,
                 aggregation=vs.VariableAggregation.NONE,
                 partitioner=None,
                 **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in Keras-style mode the synchronization and aggregation arguments are
    # silently ignored (forced to automatic and none), and trainable is combined with the
    # layer's trainable flag even when trainable is null (yielding null rather than true).
    # KNOWN ISSUE: the docstring says a runtime error is raised for partitioned variable
    # regularization in eager mode; no such check exists here. Unknown keyword arguments
    # (other than the experimental autocast one) raise a type error, which the docstring omits.
    # Creates or fetches (via variable scope reuse) a layer variable. Inputs are those listed in
    # the docstring: name, shape, dtype, initializer, regularizer, trainable, constraint,
    # use_resource, synchronization, aggregation, partitioner, plus extra keywords.
    # Keras style: delegates to the Keras base add_weight. Otherwise: validates that
    # on-read synchronization is not combined with trainable (value error), defaults trainable
    # to true, records existing graph variables so regularizers are only added for new ones,
    # defaults dtype to the layer dtype or float32, sets the scope, reuses variables if the
    # layer is built or reuse is requested, and creates the variable via the variable-scope
    # getter. Registers the regularizer (also with the variable store shim when it supports
    # that). If a custom getter overrode trainability, moves the variable from the trainable
    # to the non-trainable list. Returns the variable. Side effects: sets the current scope,
    # mutates weight lists, losses and variable collections.
    """Adds a new variable to the layer, or gets an existing one; returns it.

    Args:
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
        Note, if the current variable scope is marked as non-trainable
        then this parameter is ignored and any added variables are also
        marked as non-trainable. `trainable` defaults to `True` unless
        `synchronization` is set to `ON_READ`.
      constraint: constraint instance (callable).
      use_resource: Whether to use `ResourceVariable`.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize. If `synchronization` is set to `ON_READ`,
        `trainable` must not be set to `True`.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      partitioner: (optional) partitioner instance (callable).  If
        provided, when the requested variable is created it will be split
        into multiple partitions according to `partitioner`.  In this case,
        an instance of `PartitionedVariable` is returned.  Available
        partitioners include `tf.compat.v1.fixed_size_partitioner` and
        `tf.compat.v1.variable_axis_size_partitioner`.  For more details, see
        the documentation of `tf.compat.v1.get_variable` and the  "Variable
        Partitioners and Sharding" section of the API guide.
      **kwargs: Additional keyword arguments.

    Returns:
      The created variable.  Usually either a `Variable` or `ResourceVariable`
      instance.  If `partitioner` is not `None`, a `PartitionedVariable`
      instance is returned.

    Raises:
      RuntimeError: If called with partitioned variable regularization and
        eager execution is enabled.
      ValueError: When trainable has been set to True with synchronization
        set as `ON_READ`.
    """
    for kwarg in kwargs:
      if kwarg != 'experimental_autocast':
        raise TypeError('Unknown keyword argument:', kwarg)
    if self._keras_style:
      return super(Layer, self).add_weight(
          name=name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          trainable=trainable and self.trainable,
          constraint=constraint,
          use_resource=use_resource,
          synchronization=vs.VariableSynchronization.AUTO,
          aggregation=vs.VariableAggregation.NONE,
          partitioner=partitioner,
          **kwargs)

    if synchronization == vs.VariableSynchronization.ON_READ:
      if trainable:
        raise ValueError(
            'Synchronization value can be set to '
            'VariableSynchronization.ON_READ only for non-trainable variables. '
            'You have specified trainable=True and '
            'synchronization=VariableSynchronization.ON_READ.')
      else:
        # Set trainable to be false when variable is to be synced on read.
        trainable = False
    elif trainable is None:
      trainable = True

    def _should_add_regularizer(variable, existing_variable_set):
      # [FUNCTION DOC]
      # HELPER: nested helper of add_weight. Returns true if the variable is new, i.e. not in
      # the given set of pre-existing variables. For a partitioned (split) variable, returns
      # false if any partition is already in the set, true otherwise.
      if base_layer_utils.is_split_variable(variable):
        for var in variable:
          if var in existing_variable_set:
            return False
        return True
      else:
        return variable not in existing_variable_set

    init_graph = None
    if not context.executing_eagerly():
      default_graph = ops.get_default_graph()
      if default_graph.building_function:
        with ops.init_scope():
          # Retrieve the variables from the graph into which variables
          # will be lifted; if initialization ops will be lifted into
          # the eager context, then there is nothing to retrieve, since variable
          # collections are not supported when eager execution is enabled.
          if not context.executing_eagerly():
            init_graph = ops.get_default_graph()
            existing_variables = set(tf_variables.global_variables())
      else:
        # Initialization ops will not be lifted out of the default graph.
        init_graph = default_graph
        existing_variables = set(tf_variables.global_variables())

    if dtype is None:
      dtype = self.dtype or dtypes.float32

    self._set_scope(None)
    reuse = self.built or self._reuse
    prev_len_trainable = len(self._trainable_weights)
    with vs.variable_scope(
        self._scope, reuse=reuse, auxiliary_name_scope=False) as scope:
      self._current_scope = scope
      with backend.name_scope(self._name_scope()):  # pylint: disable=not-callable
        use_resource = (use_resource or
                        self._use_resource_variables or
                        scope.use_resource)
        if initializer is None:
          initializer = scope.initializer
        variable = super(Layer, self).add_weight(
            name,
            shape,
            dtype=dtypes.as_dtype(dtype),
            initializer=initializer,
            trainable=trainable and self.trainable,
            constraint=constraint,
            partitioner=partitioner,
            use_resource=use_resource,
            synchronization=synchronization,
            aggregation=aggregation,
            getter=vs.get_variable,
            **kwargs)

        if regularizer:
          if (ops.executing_eagerly_outside_functions()
              or _should_add_regularizer(variable, existing_variables)):
            self._handle_weight_regularization(name, variable, regularizer)
            var_store = vs._get_default_variable_store()  # pylint: disable=protected-access
            # When the shim to get variable scope working in TF2 is used,
            # We need to explicitly make the shim track the regularization
            # losses as the collections will not be accessible.
            if hasattr(var_store, 'add_regularizer'):
              var_store.add_regularizer(variable, regularizer)

        if init_graph is not None:
          # Handle edge case where a custom getter has overridden `trainable`.
          # There is one known occurrence of this, in unit test
          # testBasicRNNCellNotTrainable in
          # contrib.rnn.python.kernel_tests.core_rnn_cell_test
          with init_graph.as_default():
            trainable_variables = tf_variables.trainable_variables()
          if (trainable and self.trainable and
              variable not in trainable_variables):
            # A custom getter / variable scope overrode the trainable flag.
            extra_trainable_vars = self._trainable_weights[prev_len_trainable:]
            self._trainable_weights = self._trainable_weights[
                :prev_len_trainable]
            self._non_trainable_weights += extra_trainable_vars
    return variable

  def __call__(self, inputs, *args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a value error is raised if call returns None; this code
    # has no such check. The value error actually raised here is for passing a scope argument
    # to a Keras-style layer.
    # Invokes the layer. Inputs: inputs (tensor(s)), extra positional arguments, and keyword
    # arguments; the reserved scope keyword is removed first. Keras style: raises a value error
    # if a scope was given, else defers to the Keras base call. Legacy style: sets the scope,
    # enters a variable scope (reuse forced on once built; the reuse scope is cached unless
    # eager outside functions), stores it as current scope, passes the scope to call if call
    # accepts one, runs the Keras base call, and in graph mode adds the layer's update ops to
    # the update-ops collection. Returns the outputs.
    """Wraps `call`, applying pre- and post-processing steps.

    Args:
      inputs: input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.
        **Note**: kwarg `scope` is reserved for use by the layer.

    Returns:
      Output tensor(s).

    Note:
      - If the layer's `call` method takes a `scope` keyword argument,
        this argument will be automatically set to the current variable scope.
      - If the layer's `call` method takes a `mask` argument (as some Keras
        layers do), its default value will be set to the mask generated
        for `inputs` by the previous layer (if `input` did come from
        a layer that generated a corresponding mask, i.e. if it came from
        a Keras layer with masking support.

    Raises:
      ValueError: if the layer's `call` method returns None (an invalid value).
    """
    scope = kwargs.pop('scope', None)

    if self._keras_style:
      if scope is not None:
        raise ValueError(
            'scope argument not allowed when keras style layers are enabled, '
            'but saw: {}'.format(scope))
      return super(Layer, self).__call__(inputs, *args, **kwargs)

    self._set_scope(scope)

    if self.built:
      try:
        # Some classes which inherit from Layer do not use its constructor, so
        # rather than initializing to None we check for an AttributeError.
        scope_context_manager = self._always_reuse_variable_scope  # pylint: disable=access-member-before-definition
      except AttributeError:
        scope_context_manager = None

      if scope_context_manager is None:
        # From this point we will always set reuse=True, so create a "final"
        # variable scope with this setting. We avoid re-creating variable scopes
        # after this point as an optimization.
        scope_context_manager = vs.variable_scope(
            self._scope, reuse=True, auxiliary_name_scope=False)

        # Do not cache variable scopes if Eager mode is enabled. If Eager mode
        # is enabled then we don't want to reuse scopes because the cached scope
        # might be from a FuncGraph or Eager scope we are no longer in.
        if not ops.executing_eagerly_outside_functions():
          self._always_reuse_variable_scope = scope_context_manager
    else:
      scope_context_manager = vs.variable_scope(
          self._scope, reuse=self._reuse, auxiliary_name_scope=False)

    with scope_context_manager as scope:
      self._current_scope = scope

      try:
        call_has_scope_arg = self._call_has_scope_arg
      except AttributeError:
        self._call_fn_args = variable_scope_shim.fn_args(self.call)
        self._call_has_scope_arg = 'scope' in self._call_fn_args
        call_has_scope_arg = self._call_has_scope_arg
      if call_has_scope_arg:
        kwargs['scope'] = scope

      # Actually call layer
      outputs = super(Layer, self).__call__(inputs, *args, **kwargs)

    if not context.executing_eagerly():
      # Update global default collections.
      _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)
    return outputs

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Custom deep copy. Input: memo, the mapping of already copied objects. Creates a new
    # instance without running the constructor, registers it in the memo, then copies each
    # attribute: graph, thread-local and metrics-lock attributes are shared; scope attributes
    # are shallow copied; tensors or tensor lists are shared; everything else is deep copied.
    # Returns the new layer.
    no_copy = set(['_graph', '_thread_local', '_metrics_lock'])
    shallow_copy = set(['_scope', '_always_reuse_variable_scope'])
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      if k in no_copy:
        setattr(result, k, v)
      elif k in shallow_copy:
        setattr(result, k, copy.copy(v))
      elif base_layer.is_tensor_or_tensor_list(v):
        setattr(result, k, v)
      else:
        setattr(result, k, copy.deepcopy(v, memo))
    return result

  def __setattr__(self, value, name):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameters are named value then name, the reverse of the normal
    # attribute-name then value order; it still works because they are passed through
    # positionally, but the names are misleading.
    # Sets an attribute while bypassing the automatic dependency tracking of the parent Layer,
    # by calling the plain object-level setter of the trackable base. Returns nothing.
    # By-pass the automatic dependency tracking performed by the parent Layer.
    super(trackable.Trackable, self).__setattr__(value, name)  # pylint: disable=bad-super-call

  @property
  def _is_legacy_layer(self):
    # [FUNCTION DOC]
    # HELPER: marker property used by Keras to detect legacy tf.layers layers. Always returns
    # true; must not be overridden.
    """Used by keras to check compatibility. This should not be overridden."""
    return True


def _add_elements_to_collection(elements, collection_list):
  # [FUNCTION DOC]
  # HELPER: used by Layer.add_loss and Layer.__call__ to register tensors in graph collections.
  # Inputs: elements (an item or nested structure of items) and collection_list (a collection
  # name or nested structure of names). Both are flattened; each element is appended to each
  # named graph collection unless the same object is already there. Raises a runtime error if
  # eager execution is enabled. Returns nothing; mutates the graph collections.
  if context.executing_eagerly():
    raise RuntimeError('Using collections from Layers not supported in Eager '
                       'mode. Tried to add %s to %s' % (elements,
                                                        collection_list))
  elements = nest.flatten(elements)
  collection_list = nest.flatten(collection_list)
  for name in collection_list:
    collection = ops.get_collection_ref(name)
    collection_set = {id(e) for e in collection}
    for element in elements:
      if id(element) not in collection_set:
        collection.append(element)
