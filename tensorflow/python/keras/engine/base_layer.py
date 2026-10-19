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
# ==============================================================================
# [MODULE DOC]
# KNOWN ISSUE: The first description paragraph below overstates the module: there is no tensor
# flattening layer wrapper, no name uniquification or dtype policy lookup helper and no
# KerasHistory helper here. The actual contents are the Layer class, TensorFlowOpLayer, AddLoss,
# AddMetric, two private helpers (functional-construction-mode check and numpy/scalar to tensor
# conversion) and a re-export of the input spec class under its old name.
# Defines the core Keras Layer class from which every layer and model inherits, plus several
# small related classes and helpers (input-spec/node bookkeeping, TF op layers, an AddLoss and
# AddMetric layer, a keras-style tensor-flattening layer wrapper, and a few module-level
# utilities for name uniquification, call-context handling and dtype policy lookup).
# Exposes Layer, Layer-related call-context helpers, TensorFlowOpLayer, AddLoss, AddMetric,
# KerasHistory-related helpers, and the lazily loaded metrics module reference.
# Import-time side effects: creates a lazy loader for the metrics module and defines module
# level constants (op-layer name prefix and the tuple of types that are automatically cast);
# it also re-exports to_snake_case and is_tensor_or_tensor_list for modules that depend only
# on this one. Relates to the rest of the package: engine modules such as base_layer_utils,
# input_spec, node and keras_tensor supply bookkeeping, while training.py, functional.py and
# every concrete layer subclass build on the Layer defined here.
# pylint: disable=protected-access
"""Contains the base Layer class, from which all layers inherit."""

import collections
import copy
import functools
import itertools
import threading
import warnings
import weakref

import numpy as np

from google.protobuf import json_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.tools.docs import doc_controls

# A module that only depends on `keras.layers` import these from here.

# pylint: disable=g-inconsistent-quotes
metrics_mod = generic_utils.LazyLoader(
    "metrics_mod", globals(),
    "tensorflow.python.keras.metrics")
# pylint: enable=g-inconsistent-quotes

# Prefix that is added to the TF op layer names.
_TF_OP_LAYER_NAME_PREFIX = 'tf_op_layer_'

# TODO(mdan): Should we have a single generic type for types that can be passed
# to tf.cast?
_AUTOCAST_TYPES = (tensor_lib.Tensor, sparse_tensor.SparseTensor,
                   ragged_tensor.RaggedTensor)


class Layer(module.Module, version_utils.LayerVersionSelector):
  # [CLASS DOC]
  # Base class of all Keras layers. A layer is a callable object holding state (weights, losses,
  # metrics, updates, dtype policy, input spec, inbound and outbound graph nodes) and a
  # computation defined by an overridable call method. Instances are created with optional
  # trainable flag, name, dtype or dtype policy, dynamic flag and a restricted set of keyword
  # arguments (input shape hints, activity regularizer, initial weights, autocast switch).
  # Applying an instance to inputs goes through the call wrapper which builds the layer lazily,
  # casts inputs, handles masks, functional-API symbolic inputs, name scopes, autograph and
  # mixed-precision loss/activity handling. Inherits from the tracking module class (for
  # variable/submodule tracking) and the version selector (which picks the v1 or v2 class).
  # Subclasses normally override build, call, get_config, compute_output_shape and
  # compute_mask. Thread-local state holds per-thread call context information.
  """This is the class from which all layers inherit.

  A layer is a callable object that takes as input one or more tensors and
  that outputs one or more tensors. It involves *computation*, defined
  in the `call()` method, and a *state* (weight variables), defined
  either in the constructor `__init__()` or in the `build()` method.

  Users will just instantiate a layer and then treat it as a callable.

  Args:
    trainable: Boolean, whether the layer's variables should be trainable.
    name: String name of the layer.
    dtype: The dtype of the layer's computations and weights. Can also be a
      `tf.keras.mixed_precision.Policy`, which allows the computation and weight
      dtype to differ. Default of `None` means to use
      `tf.keras.mixed_precision.global_policy()`, which is a float32 policy
      unless set to different value.
    dynamic: Set this to `True` if your layer should only be run eagerly, and
      should not be used to generate a static computation graph.
      This would be the case for a Tree-RNN or a recursive network,
      for example, or generally for any layer that manipulates tensors
      using Python control flow. If `False`, we assume that the layer can
      safely be used to generate a static computation graph.

  Attributes:
    name: The name of the layer (string).
    dtype: The dtype of the layer's weights.
    variable_dtype: Alias of `dtype`.
    compute_dtype: The dtype of the layer's computations. Layers automatically
      cast inputs to this dtype which causes the computations and output to also
      be in this dtype. When mixed precision is used with a
      `tf.keras.mixed_precision.Policy`, this will be different than
      `variable_dtype`.
    dtype_policy: The layer's dtype policy. See the
      `tf.keras.mixed_precision.Policy` documentation for details.
    trainable_weights: List of variables to be included in backprop.
    non_trainable_weights: List of variables that should not be
      included in backprop.
    weights: The concatenation of the lists trainable_weights and
      non_trainable_weights (in this order).
    trainable: Whether the layer should be trained (boolean), i.e. whether
      its potentially-trainable weights should be returned as part of
      `layer.trainable_weights`.
    input_spec: Optional (list of) `InputSpec` object(s) specifying the
      constraints on inputs that can be accepted by the layer.

  We recommend that descendants of `Layer` implement the following methods:

  * `__init__()`: Defines custom layer attributes, and creates layer state
    variables that do not depend on input shapes, using `add_weight()`.
  * `build(self, input_shape)`: This method can be used to create weights that
    depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
    will automatically build the layer (if it has not been built yet) by
    calling `build()`.
  * `call(self, inputs, *args, **kwargs)`: Called in `__call__` after making
    sure `build()` has been called. `call()` performs the logic of applying the
    layer to the input tensors (which should be passed in as argument).
    Two reserved keyword arguments you can optionally use in `call()` are:
      - `training` (boolean, whether the call is in inference mode or training
        mode). See more details in [the layer/model subclassing guide](
        https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method)
      - `mask` (boolean tensor encoding masked timesteps in the input, used
        in RNN layers). See more details in [the layer/model subclassing guide](
        https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_mask_argument_in_the_call_method)
    A typical signature for this method is `call(self, inputs)`, and user could
    optionally add `training` and `mask` if the layer need them. `*args` and
    `**kwargs` is only useful for future extension when more input parameters
    are planned to be added.
  * `get_config(self)`: Returns a dictionary containing the configuration used
    to initialize this layer. If the keys differ from the arguments
    in `__init__`, then override `from_config(self)` as well.
    This method is used when saving
    the layer or a model that contains this layer.

  Examples:

  Here's a basic example: a layer with two variables, `w` and `b`,
  that returns `y = w . x + b`.
  It shows how to implement `build()` and `call()`.
  Variables set as attributes of a layer are tracked as weights
  of the layers (in `layer.weights`).

  ```python
  class SimpleDense(Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
      w_init = tf.random_normal_initializer()
      self.w = tf.Variable(
          initial_value=w_init(shape=(input_shape[-1], self.units),
                               dtype='float32'),
          trainable=True)
      b_init = tf.zeros_initializer()
      self.b = tf.Variable(
          initial_value=b_init(shape=(self.units,), dtype='float32'),
          trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b

  # Instantiates the layer.
  linear_layer = SimpleDense(4)

  # This will also call `build(input_shape)` and create the weights.
  y = linear_layer(tf.ones((2, 2)))
  assert len(linear_layer.weights) == 2

  # These weights are trainable, so they're listed in `trainable_weights`:
  assert len(linear_layer.trainable_weights) == 2
  ```

  Note that the method `add_weight()` offers a shortcut to create weights:

  ```python
  class SimpleDense(Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
  ```

  Besides trainable weights, updated via backpropagation during training,
  layers can also have non-trainable weights. These weights are meant to
  be updated manually during `call()`. Here's a example layer that computes
  the running sum of its inputs:

  ```python
  class ComputeSum(Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight.
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                 trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

  my_sum = ComputeSum(2)
  x = tf.ones((2, 2))

  y = my_sum(x)
  print(y.numpy())  # [2. 2.]

  y = my_sum(x)
  print(y.numpy())  # [4. 4.]

  assert my_sum.weights == [my_sum.total]
  assert my_sum.non_trainable_weights == [my_sum.total]
  assert my_sum.trainable_weights == []
  ```

  For more information about creating layers, see the guide
  [Making new Layers and Models via subclassing](
    https://www.tensorflow.org/guide/keras/custom_layers_and_models)
  """

  # See tf.Module for the usage of this property.
  # The key for _obj_reference_counts_dict is a Trackable, which could be a
  # variable or layer etc. tf.Module._flatten will fail to flatten the key
  # since it is trying to convert Trackable to a string. This attribute can be
  # ignored even after the fix of nest lib, since the trackable object should
  # already been available as individual attributes. _obj_reference_counts_dict
  # just contains a copy of them.
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_obj_reference_counts_dict',),
      module.Module._TF_MODULE_IGNORED_PROPERTIES
  ))

  # When loading from a SavedModel, Layers typically can be revived into a
  # generic Layer wrapper. Sometimes, however, layers may implement methods
  # that go beyond this wrapper, as in the case of PreprocessingLayers'
  # `adapt` method. When this is the case, layer implementers can override
  # must_restore_from_config to return True; layers with this property must
  # be restored into their actual objects (and will fail if the object is
  # not available to the restoration code).
  _must_restore_from_config = False

  def _get_cell_name(self):
    # [FUNCTION DOC]
    # HELPER: Layer usage instrumentation (builds the label used to record which layer class is used).
    # Returns a string naming this layer's class: the public canonical API name with a tf. prefix
    # when the class is exported under the keras API, otherwise the fully qualified module and
    # class name. Takes no arguments and has no side effects.
    canonical_name = get_canonical_name_for_symbol(
        self.__class__, api_name='keras', add_prefix_to_v1_names=True)
    if canonical_name is not None:
      return 'tf.{}'.format(canonical_name)
    return self.__class__.__module__ + '.' + self.__class__.__name__

  def _instrument_layer_creation(self):
    # [FUNCTION DOC]
    # HELPER: Called from the constructor to set instrumentation flags.
    # Sets three boolean attributes on the instance recording whether this object counts as an
    # instrumented keras API object, and whether it is a layer class or a model class. All are
    # false when the instance has a true disable-instrumentation attribute; otherwise the api flag
    # is true and exactly one of the model/layer flags is true depending on whether the instance
    # has a true is-model-for-instrumentation attribute. Returns nothing.
    self._instrumented_keras_api = False
    self._instrumented_keras_layer_class = False
    self._instrumented_keras_model_class = False
    if not getattr(self, '_disable_keras_instrumentation', False):
      self._instrumented_keras_api = True
      if getattr(self, '_is_model_for_instrumentation', False):
        self._instrumented_keras_model_class = True
      else:
        self._instrumented_keras_layer_class = True

  @trackable.no_automatic_dependency_tracking
  def __init__(self,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor of every layer. Inputs: trainable (bool, default true) whether weights are
    # trained; name (optional string, auto-generated from the class name if missing); dtype
    # (optional dtype or dtype policy, default from the global policy); dynamic (bool, default
    # false) whether the layer must run eagerly only; and keyword arguments restricted to input
    # dim, input shape, batch input shape, batch size, initial weights, activity regularizer,
    # autocast and implementation. Unknown keywords make the keyword validator raise an error.
    # Side effects: initializes all bookkeeping state (weight lists, losses, metrics, updates,
    # thread-local storage, metrics lock, tracked trackables, inbound/outbound node lists, input
    # spec, call-function argument introspection, dtype policy, autocast flag, saved-model specs),
    # translates an input dim into an input shape, and records a batch input shape if input
    # hints were given so that an input layer can be inserted later. Stores requested initial
    # weights, enables sub-layer auto tracking and disables input-structure preservation.
    # The supports-masking flag is derived from whether compute_mask is overridden. Returns
    # nothing. Runs without automatic dependency tracking.
    self._instrument_layer_creation()

    # These properties should be set by the user via keyword arguments.
    # note that 'dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {
        'input_dim',
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'weights',
        'activity_regularizer',
        'autocast',
        'implementation',
    }
    # Validate optional keyword arguments.
    generic_utils.validate_kwargs(kwargs, allowed_kwargs)

    # Mutable properties
    # Indicates whether the layer's weights are updated during training
    # and whether the layer's updates are run during training.
    self._trainable = trainable
    # A stateful layer is a layer whose updates are run during inference too,
    # for instance stateful RNNs.
    self._stateful = False
    # Indicates whether `build` needs to be called upon layer call, to create
    # the layer's weights.
    self.built = False
    # Provides information about which inputs are compatible with the layer.
    self._input_spec = None

    # SavedModel-related attributes.
    # Record the build input shape for loading purposes.
    # TODO(kathywu): Move this to Layer._set_save_spec once cl/290121460 is
    # submitted.
    self._build_input_shape = None
    self._saved_model_inputs_spec = None

    # `Layer.compute_mask` will be called at the end of `Layer.__call__` if
    # `Layer.compute_mask` is overridden, or if the `Layer` subclass sets
    # `self.supports_masking=True`.
    self._supports_masking = not generic_utils.is_default(self.compute_mask)

    self._init_set_name(name)
    self._activity_regularizer = regularizers.get(
        kwargs.pop('activity_regularizer', None))
    self._maybe_create_attribute('_trainable_weights', [])
    self._maybe_create_attribute('_non_trainable_weights', [])
    self._updates = []
    # Object to store all thread local layer properties.
    self._thread_local = threading.local()
    # A list of zero-argument lambdas which return Tensors, used for variable
    # regularizers.
    self._callable_losses = []
    # A list of symbolic Tensors containing activity regularizers and losses
    # manually added through `add_loss` in graph-building mode.
    self._losses = []
    # A list of metric instances corresponding to the symbolic metric tensors
    # added using the `add_metric` API.
    self._metrics = []
    # Ensures the same metric is not added multiple times in `MirroredStrategy`.
    self._metrics_lock = threading.Lock()

    # Both graph and subclassed networks have a dtype policy. For graph
    # networks, the policy's compute and variable dtypes are ignored. Such
    # networks only use the policy if it is a PolicyV1, in which case it uses
    # the PolicyV1's loss_scale (Policy does not have a loss_scale). For
    # subclassed networks, the compute and variable dtypes are used as like any
    # ordinary layer.
    self._set_dtype_policy(dtype)
    # Boolean indicating whether the layer automatically casts its inputs to the
    # layer's compute_dtype.
    self._autocast = kwargs.get('autocast',
                                base_layer_utils.v2_dtype_behavior_enabled())

    # Tracks `TrackableDataStructure`s, `Module`s, and `Layer`s.
    # Ordered by when the object was assigned as an attr.
    # Entries are unique.
    self._maybe_create_attribute('_self_tracked_trackables', [])

    # These lists will be filled via successive calls
    # to self._add_inbound_node().
    # Used in symbolic mode only, only in conjunction with graph-networks
    self._inbound_nodes_value = []
    self._outbound_nodes_value = []

    self._init_call_fn_args()

    # Whether the `call` method can be used to build a TF graph without issues.
    # This attribute has no effect if the model is created using the Functional
    # API. Instead, `model.dynamic` is determined based on the internal layers.
    self._dynamic = dynamic

    # Manage input shape information if passed.
    if 'input_dim' in kwargs and 'input_shape' not in kwargs:
      # Backwards compatibility: alias 'input_dim' to 'input_shape'.
      kwargs['input_shape'] = (kwargs['input_dim'],)
    if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
      # In this case we will later create an input layer
      # to insert before the current layer
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        if 'batch_size' in kwargs:
          batch_size = kwargs['batch_size']
        else:
          batch_size = None
        batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
      self._batch_input_shape = batch_input_shape

    # Manage initial weight values if passed.
    self._initial_weights = kwargs.get('weights', None)

    # Whether the layer will track any layers that is set as attribute on itself
    # as sub-layers, the weights from the sub-layers will be included in the
    # parent layer's variables() as well.
    # Default to True, which means auto tracking is turned on. Certain subclass
    # might want to turn it off, like Sequential model.
    self._auto_track_sub_layers = True

    # For backwards compat reasons, most built-in layers do not guarantee
    # That they will 100% preserve the structure of input args when saving
    # / loading configs. E.g. they may un-nest an arg that is
    # a list with one element.
    self._preserve_input_structure_in_config = False

  @trackable.no_automatic_dependency_tracking
  @generic_utils.default
  def build(self, input_shape):
    # [FUNCTION DOC]
    # Default (no-op) weight-creation hook, meant to be overridden. Input: input_shape, a tensor
    # shape or list of shapes (one per input). If this method has been overridden by a subclass
    # (it is not the marked default), the shape is stored as the build input shape used for
    # saving. In every case the built flag is set to true. Returns nothing.
    """Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Args:
      input_shape: Instance of `TensorShape`, or list of instances of
        `TensorShape` if the layer expects a list of inputs
        (one instance per input).
    """
    # Only record the build input shapes of overridden build methods.
    if not hasattr(self.build, '_is_default'):
      self._build_input_shape = input_shape
    self.built = True

  @doc_controls.for_subclass_implementers
  def call(self, inputs, *args, **kwargs):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Default forward computation, meant to be overridden by subclasses. Inputs: inputs (a tensor
    # or nested structure of tensors), extra positional and keyword arguments which are ignored
    # here (reserved keywords for overriding implementations are a training flag and a mask).
    # Returns the inputs unchanged (identity). No side effects.
    """This is where the layer's logic lives.

    Note here that `call()` method in `tf.keras` is little bit different
    from `keras` API. In `keras` API, you can pass support masking for
    layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
    method to support masking.

    Args:
      inputs: Input tensor, or dict/list/tuple of input tensors.
        The first positional `inputs` argument is subject to special rules:
        - `inputs` must be explicitly passed. A layer cannot have zero
          arguments, and `inputs` cannot be provided via the default value
          of a keyword argument.
        - NumPy array or Python scalar values in `inputs` get cast as tensors.
        - Keras mask metadata is only collected from `inputs`.
        - Layers are built (`build(input_shape)` method)
          using shape info from `inputs` only.
        - `input_spec` compatibility is only checked against `inputs`.
        - Mixed precision input casting is only applied to `inputs`.
          If a layer has tensor arguments in `*args` or `**kwargs`, their
          casting behavior in mixed precision should be handled manually.
        - The SavedModel input specification is generated using `inputs` only.
        - Integration with various ecosystem packages like TFMOT, TFLite,
          TF.js, etc is only supported for `inputs` and not for tensors in
          positional and keyword arguments.
      *args: Additional positional arguments. May contain tensors, although
        this is not recommended, for the reasons above.
      **kwargs: Additional keyword arguments. May contain tensors, although
        this is not recommended, for the reasons above.
        The following optional keyword arguments are reserved:
        - `training`: Boolean scalar tensor of Python boolean indicating
          whether the `call` is meant for training or inference.
        - `mask`: Boolean input mask. If the layer's `call()` method takes a
          `mask` argument, its default value will be set to the mask generated
          for `inputs` by the previous layer (if `input` did come from a layer
          that generated a corresponding mask, i.e. if it came from a Keras
          layer with masking support).

    Returns:
      A tensor or list/tuple of tensors.
    """
    return inputs

  @doc_controls.for_subclass_implementers
  def _add_trackable(self, trackable_object, trainable):
    # [FUNCTION DOC]
    # Registers a trackable object (for example a lookup table or other non-variable state) so its
    # weights are reported by the layer. Inputs: trackable_object, either an already wrapped weight
    # handler or any trackable to wrap in one; trainable, bool choosing whether the handler is
    # appended to the trainable or non-trainable weight list. Returns the weight handler used.
    # Side effect: mutates the layer's weight lists.
    """Adds a Trackable object to this layer's state.

    Args:
      trackable_object: The tf.tracking.Trackable object to add.
      trainable: Boolean, whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases) or
        "non_trainable_variables" (e.g. BatchNorm mean and variance).

    Returns:
      The TrackableWeightHandler used to track this object.
    """
    if isinstance(trackable_object, base_layer_utils.TrackableWeightHandler):
      handler = trackable_object
    else:
      handler = base_layer_utils.TrackableWeightHandler(trackable_object)
    if trainable:
      self._trainable_weights.append(handler)
    else:
      self._non_trainable_weights.append(handler)
    return handler

  @doc_controls.for_subclass_implementers
  def add_weight(self,
                 name=None,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 use_resource=None,
                 synchronization=tf_variables.VariableSynchronization.AUTO,
                 aggregation=tf_variables.VariableAggregation.NONE,
                 **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring lists only a value error under Raises, but the code also raises a
    # type error for any unrecognized extra keyword (only getter, collections, experimental
    # autocast and caching device are allowed; a partitioner keyword is silently ignored). It
    # also raises a value error when no initializer is given for a dtype that is not floating,
    # integer, unsigned or boolean (unless a custom getter is supplied).
    # Creates, tracks and returns a new weight variable. Inputs: name, shape (default scalar),
    # dtype (default layer dtype or the backend float type), initializer, regularizer and
    # constraint (each resolved from identifiers or callables), trainable (default true; forced
    # false for on-read synchronization, and true with on-read synchronization raises a value
    # error), use_resource, synchronization and aggregation modes, and the extra keywords above.
    # With no initializer, float dtypes get glorot-uniform and integer/bool dtypes get zeros.
    # If the dtype policy is the inferred one it is replaced by a policy named after the dtype.
    # Under a mixed-precision policy, floating variables are wrapped as auto-cast variables and
    # a caching device is dropped with a warning. The variable is created via the tracking
    # custom-getter mechanism, a regularization loss is attached if a regularizer was given,
    # the variable is registered with the backend, and appended (per partition if split) to the
    # trainable or non-trainable list. Returns the created variable.
    """Adds a new variable to the layer.

    Args:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype`.
      initializer: Initializer instance (callable).
      regularizer: Regularizer instance (callable).
      trainable: Boolean, whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean and variance).
        Note that `trainable` cannot be `True` if `synchronization`
        is set to `ON_READ`.
      constraint: Constraint instance (callable).
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
      **kwargs: Additional keyword arguments. Accepted values are `getter`,
        `collections`, `experimental_autocast` and `caching_device`.

    Returns:
      The variable created.

    Raises:
      ValueError: When giving unsupported dtype and no initializer or when
        trainable has been set to True with synchronization set as `ON_READ`.
    """
    if shape is None:
      shape = ()
    kwargs.pop('partitioner', None)  # Ignored.
    # Validate optional keyword arguments.
    for kwarg in kwargs:
      if kwarg not in ['collections', 'experimental_autocast',
                       'caching_device', 'getter']:
        raise TypeError('Unknown keyword argument:', kwarg)
    collections_arg = kwargs.pop('collections', None)
    # 'experimental_autocast' can be set to False by the caller to indicate an
    # AutoCastVariable should never be created.
    autocast = kwargs.pop('experimental_autocast', True)
    # See the docstring for tf.Variable about the details for caching_device.
    caching_device = kwargs.pop('caching_device', None)

    if dtype is None:
      dtype = self.dtype or backend.floatx()
    dtype = dtypes.as_dtype(dtype)
    if self._dtype_policy.variable_dtype is None:
      # The policy is "_infer", so we infer the policy from the variable dtype.
      self._set_dtype_policy(policy.Policy(dtype.base_dtype.name))
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    constraint = constraints.get(constraint)

    if synchronization == tf_variables.VariableSynchronization.ON_READ:
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

    # Initialize variable when no initializer provided
    if initializer is None:
      # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
      if dtype.is_floating:
        initializer = initializers.get('glorot_uniform')
      # If dtype is DT_INT/DT_UINT, provide a default value `zero`
      # If dtype is DT_BOOL, provide a default value `FALSE`
      elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
        initializer = initializers.get('zeros')
      # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
      elif 'getter' not in kwargs:
        # When `getter` is specified, it's possibly fine for `initializer` to be
        # None since it's up to the custom `getter` to raise error in case it
        # indeed needs `initializer`.
        raise ValueError('An initializer for variable %s of type %s is required'
                         ' for layer %s' % (name, dtype.base_dtype, self.name))

    getter = kwargs.pop('getter', base_layer_utils.make_variable)
    if (autocast and
        self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype
        and dtype.is_floating):
      old_getter = getter
      # Wrap variable constructor to return an AutoCastVariable.
      def getter(*args, **kwargs):  # pylint: disable=function-redefined
        # [FUNCTION DOC]
        # HELPER: Variable-creation wrapper defined inside add_weight for mixed-precision policies.
        # Takes any positional and keyword arguments, forwards them to the original getter to create
        # the variable, wraps the result as an auto-cast variable and returns it.
        variable = old_getter(*args, **kwargs)
        return autocast_variable.create_autocast_variable(variable)
      # Also the caching_device does not work with the mixed precision API,
      # disable it if it is specified.
      # TODO(b/142020079): Reenable it once the bug is fixed.
      if caching_device is not None:
        tf_logging.warning(
            '`caching_device` does not work with mixed precision API. Ignoring '
            'user specified `caching_device`.')
        caching_device = None

    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        # TODO(allenl): a `make_variable` equivalent should be added as a
        # `Trackable` method.
        getter=getter,
        # Manage errors in Layer rather than Trackable.
        overwrite=True,
        initializer=initializer,
        dtype=dtype,
        constraint=constraint,
        trainable=trainable,
        use_resource=use_resource,
        collections=collections_arg,
        synchronization=synchronization,
        aggregation=aggregation,
        caching_device=caching_device)
    if regularizer is not None:
      # TODO(fchollet): in the future, this should be handled at the
      # level of variable creation, and weight regularization losses
      # should be variable attributes.
      name_in_scope = variable.name[:variable.name.find(':')]
      self._handle_weight_regularization(name_in_scope,
                                         variable,
                                         regularizer)
    if base_layer_utils.is_split_variable(variable):
      for v in variable:
        backend.track_variable(v)
        if trainable:
          self._trainable_weights.append(v)
        else:
          self._non_trainable_weights.append(v)
    else:
      backend.track_variable(variable)
      if trainable:
        self._trainable_weights.append(variable)
      else:
        self._non_trainable_weights.append(variable)
    return variable

  @generic_utils.default
  def get_config(self):
    # [FUNCTION DOC]
    # Default serialization of the layer configuration. Takes no arguments. Returns a dictionary
    # with the name, trainable flag, serialized dtype policy, the batch input shape if one was
    # recorded, and the dynamic flag only when it is true. Raises a not-implemented error if the
    # constructor takes arguments beyond the ones in the config and the method was not overridden
    # (so subclasses with extra constructor arguments must override it). Not a fresh copy
    # guarantee: callers should copy before modifying. No side effects.
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable)
    containing the configuration of a layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    The config of a layer does not include connectivity
    information, nor the layer class name. These are handled
    by `Network` (one layer of abstraction above).

    Note that `get_config()` does not guarantee to return a fresh copy of dict
    every time it is called. The callers should make a copy of the returned dict
    if they want to modify it.

    Returns:
        Python dictionary.
    """
    all_args = tf_inspect.getfullargspec(self.__init__).args
    config = {
        'name': self.name,
        'trainable': self.trainable,
    }
    if hasattr(self, '_batch_input_shape'):
      config['batch_input_shape'] = self._batch_input_shape
    config['dtype'] = policy.serialize(self._dtype_policy)
    if hasattr(self, 'dynamic'):
      # Only include `dynamic` in the `config` if it is `True`
      if self.dynamic:
        config['dynamic'] = self.dynamic
      elif 'dynamic' in all_args:
        all_args.remove('dynamic')
    expected_args = config.keys()
    # Finds all arguments in the `__init__` that are not in the config:
    extra_args = [arg for arg in all_args if arg not in expected_args]
    # Check that either the only argument in the `__init__` is  `self`,
    # or that `get_config` has been overridden:
    if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
      raise NotImplementedError('Layer %s has arguments in `__init__` and '
                                'therefore must override `get_config`.' %
                                self.__class__.__name__)
    return config

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method that rebuilds a layer from a configuration mapping (usually the output of
    # get_config) by passing every entry to the constructor as a keyword argument. Returns the new
    # layer instance. Does not restore weights or connectivity. Errors from the constructor
    # (for example unexpected keys) propagate.
    """Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Args:
        config: A Python dictionary, typically the
            output of get_config.

    Returns:
        A layer instance.
    """
    return cls(**config)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it returns an input shape tuple; the code actually returns a
    # (nested structure of) tensor shape objects describing the layer OUTPUT, obtained by running
    # the layer on placeholders in a scratch graph. It also does not mention that it raises a
    # not-implemented error when not executing eagerly or when shape inference fails.
    # Default shape inference. Input: input_shape, a shape or nested structure of shapes.
    # In eager mode it builds the layer if needed (side effect), creates placeholder inputs in a
    # temporary function graph, calls the layer in inference mode and returns the shapes of the
    # outputs. A type error during that call is converted into a not-implemented error asking
    # the user to implement this method. Outside eager mode it raises a not-implemented error.
    """Computes the output shape of the layer.

    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.

    Args:
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.

    Returns:
        An input shape tuple.
    """
    if context.executing_eagerly():
      # In this case we build the model first in order to do shape inference.
      # This is acceptable because the framework only calls
      # `compute_output_shape` on shape values that the layer would later be
      # built for. It would however cause issues in case a user attempts to
      # use `compute_output_shape` manually with shapes that are incompatible
      # with the shape the Layer will be called on (these users will have to
      # implement `compute_output_shape` themselves).
      self._maybe_build(input_shape)
      with func_graph.FuncGraph(str(self.name) + '_scratch_graph').as_default():
        input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
        def _make_placeholder_like(shape):
          # [FUNCTION DOC]
          # HELPER: Used by compute_output_shape to create dummy inputs.
          # Creates a backend placeholder with the given shape and the layer's dtype, clears its
          # mask attribute (sets it to nothing) and returns it.
          ph = backend.placeholder(shape=shape, dtype=self.dtype)
          ph._keras_mask = None
          return ph
        inputs = nest.map_structure(_make_placeholder_like, input_shape)
        try:
          outputs = self(inputs, training=False)
        except TypeError as e:
          raise NotImplementedError(
              'We could not automatically infer the static shape of the '
              'layer\'s output. Please implement the '
              '`compute_output_shape` method on your layer (%s).' %
              self.__class__.__name__) from e
      return nest.map_structure(lambda t: t.shape, outputs)
    raise NotImplementedError(
        'Please run in eager mode or implement the `compute_output_shape` '
        'method on your layer (%s).' % self.__class__.__name__)

  @doc_controls.for_subclass_implementers
  def compute_output_signature(self, input_signature):
    # [FUNCTION DOC]
    # Computes the output tensor specs of the layer. Input: input_signature, a tensor spec or
    # nested structure of tensor specs. It extracts the shapes, calls compute_output_shape, and
    # returns a structure of tensor specs with those output shapes and the layer's compute dtype
    # (falling back to the first input's dtype when the layer has no dtype). Raises a type error
    # if any entry is not a tensor spec. No other side effects beyond those of compute_output_shape.
    """Compute the output tensor signature of the layer based on the inputs.

    Unlike a TensorShape object, a TensorSpec object contains both shape
    and dtype information for a tensor. This method allows layers to provide
    output dtype information if it is different from the input dtype.
    For any layer that doesn't implement this function,
    the framework will fall back to use `compute_output_shape`, and will
    assume that the output dtype matches the input dtype.

    Args:
      input_signature: Single TensorSpec or nested structure of TensorSpec
        objects, describing a candidate input for the layer.

    Returns:
      Single TensorSpec or nested structure of TensorSpec objects, describing
        how the layer would transform the provided input.

    Raises:
      TypeError: If input_signature contains a non-TensorSpec object.
    """
    def check_type_return_shape(s):
      # [FUNCTION DOC]
      # HELPER: Validation callback for compute_output_signature.
      # Returns the shape of the given entry, which must be a tensor spec; raises a type error
      # naming the offending entry otherwise.
      if not isinstance(s, tensor_lib.TensorSpec):
        raise TypeError('Only TensorSpec signature types are supported, '
                        'but saw signature entry: {}.'.format(s))
      return s.shape
    input_shape = nest.map_structure(check_type_return_shape, input_signature)
    output_shape = self.compute_output_shape(input_shape)
    dtype = self._compute_dtype
    if dtype is None:
      input_dtypes = [s.dtype for s in nest.flatten(input_signature)]
      # Default behavior when self.dtype is None, is to use the first input's
      # dtype.
      dtype = input_dtypes[0]
    return nest.map_structure(
        lambda s: tensor_lib.TensorSpec(dtype=dtype, shape=s),
        output_shape)

  def _keras_tensor_symbolic_call(self, inputs, input_masks, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: Functional-API construction glue used by the functional construction call.
    # Produces symbolic outputs for a call on symbolic inputs. Inputs: inputs (structure of
    # symbolic tensors), input_masks (their masks), args and kwargs (extra call arguments).
    # For a dynamic layer it derives tensor specs from the inputs, asks compute_output_signature
    # and wraps each result in a symbolic keras tensor without running the call. Otherwise it
    # delegates to output-signature inference, which runs call on placeholders. Returns the
    # structure of symbolic output tensors.
    if self.dynamic:
      # We will use static shape inference to return symbolic tensors
      # matching the specifications of the layer outputs.
      # Since `self.dynamic` is True, we will never attempt to
      # run the underlying TF graph (which is disconnected).
      # TODO(fchollet): consider py_func as an alternative, which
      # would enable us to run the underlying graph if needed.
      input_signature = nest.map_structure(
          lambda x: tensor_lib.TensorSpec(shape=x.shape, dtype=x.dtype),
          inputs)
      output_signature = self.compute_output_signature(input_signature)
      return nest.map_structure(keras_tensor.KerasTensor, output_signature)
    else:
      return self._infer_output_signature(inputs, args, kwargs, input_masks)

  def _infer_output_signature(self, inputs, args, kwargs, input_masks):
    # [FUNCTION DOC]
    # HELPER: Functional-API construction glue (shape/dtype inference by tracing the layer).
    # KNOWN ISSUE: The existing docstring is only a placeholder TODO and describes nothing.
    # Runs the layer's call on placeholder versions of the symbolic inputs inside a scratch
    # function graph and returns symbolic keras tensors matching the outputs. Inputs: inputs,
    # args, kwargs and input_masks, all possibly containing symbolic keras tensors that are
    # converted to placeholders. The call function is autograph-converted for subclassed layers
    # that were not loaded from a saved model. Within the scratch graph it enters the name scope
    # and auto-cast context, builds the layer if needed (side effect: creates weights), casts
    # inputs, calls the layer, handles activity regularization and sets mask metadata without
    # building a graph. If the layer has a set-inputs method and no inputs recorded yet (a
    # subclassed network), it records the inputs and outputs. Returns the symbolic outputs.
    """TODO(kaftan): Docstring."""

    call_fn = self.call
    # Wrapping `call` function in autograph to allow for dynamic control
    # flow and control dependencies in call. We are limiting this to
    # subclassed layers as autograph is strictly needed only for
    # subclassed layers and models.
    # tf_convert will respect the value of autograph setting in the
    # enclosing tf.function, if any.
    if (base_layer_utils.is_subclassed(self) and
        not base_layer_utils.from_saved_model(self)):
      call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())

    # We enter a scratch graph and build placeholder inputs inside of it that
    # match the input args.
    # We then call the layer inside of the scratch graph to identify the
    # output signatures, then we build KerasTensors corresponding to those
    # outputs.
    scratch_graph = func_graph.FuncGraph(str(self.name) + '_scratch_graph')
    with scratch_graph.as_default():
      inputs = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, inputs)
      args = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, args)
      kwargs = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, kwargs)
      input_masks = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, input_masks)

      with backend.name_scope(self._name_scope()):  # pylint: disable=not-callable
        with autocast_variable.enable_auto_cast_variables(
            self._compute_dtype_object):
          # Build layer if applicable (if the `build` method has been
          # overridden).
          # TODO(kaftan): do we maybe_build here, or have we already done it?
          self._maybe_build(inputs)
          inputs = self._maybe_cast_inputs(inputs)
          outputs = call_fn(inputs, *args, **kwargs)

        self._handle_activity_regularization(inputs, outputs)
      self._set_mask_metadata(inputs, outputs, input_masks,
                              build_graph=False)
      outputs = nest.map_structure(
          keras_tensor.keras_tensor_from_tensor, outputs)

    if hasattr(self, '_set_inputs') and not self.inputs:
      # TODO(kaftan): figure out if we need to do this at all
      # Subclassed network: explicitly set metadata normally set by
      # a call to self._set_inputs().
      self._set_inputs(inputs, outputs)
    del scratch_graph
    return outputs

  @generic_utils.default
  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Default mask propagation. Inputs: inputs (tensor or list; unused) and mask (tensor, list
    # or nothing, default nothing). If the layer does not support masking, it raises a type
    # error when any provided mask is not nothing, and otherwise returns nothing. If the layer
    # supports masking it returns the incoming mask unchanged. No side effects.
    """Computes an output mask tensor.

    Args:
        inputs: Tensor or list of tensors.
        mask: Tensor or list of tensors.

    Returns:
        None or a tensor (or list of tensors,
            one per output tensor of the layer).
    """
    if not self._supports_masking:
      if any(m is not None for m in nest.flatten(mask)):
        raise TypeError('Layer ' + self.name + ' does not support masking, '
                        'but was passed an input_mask: ' + str(mask))
      # masking not explicitly supported: return None as mask.
      return None
    # if masking is explicitly supported, by default
    # carry over the input mask
    return mask

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Applies the layer to inputs. The first positional argument is the inputs (tensor, nested
    # structure, numpy array or scalar); remaining positional and keyword arguments are passed on
    # to call. Steps: verifies the constructor chain ran (raises a runtime error if the thread-local
    # storage is missing); if the inputs are symbolic keras tensors it takes the functional
    # construction path; otherwise converts numpy values and Python numbers to tensors, resolves
    # the input masks (explicit mask argument beats implicit mask attribute), resolves the
    # training mode (explicit argument, outer call context, global learning phase, call default,
    # else inference), clears losses on the outermost call, enters the call context, checks
    # the input spec, picks eager or autograph-converted call and a name scope, builds the layer
    # if needed, auto-casts inputs, runs call under auto-cast variable context, handles activity
    # regularization, sets output mask metadata when masking is supported, and records the saved
    # model input spec on the first call. Returns the outputs of call.
    # Side effects: may build weights, update losses, call-context and mask attributes.
    """Wraps `call`, applying pre- and post-processing steps.

    Args:
      *args: Positional arguments to be passed to `self.call`.
      **kwargs: Keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).

    Note:
      - The following optional keyword arguments are reserved for specific uses:
        * `training`: Boolean scalar tensor of Python boolean indicating
          whether the `call` is meant for training or inference.
        * `mask`: Boolean input mask.
      - If the layer's `call` method takes a `mask` argument (as some Keras
        layers do), its default value will be set to the mask generated
        for `inputs` by the previous layer (if `input` did come from
        a layer that generated a corresponding mask, i.e. if it came from
        a Keras layer with masking support.
      - If the layer is not built, the method will call `build`.

    Raises:
      ValueError: if the layer's `call` method returns None (an invalid value).
      RuntimeError: if `super().__init__()` was not called in the constructor.
    """
    if not hasattr(self, '_thread_local'):
      raise RuntimeError(
          'You must call `super().__init__()` in the layer constructor.')

    # `inputs` (the first arg in the method spec) is special cased in
    # layer call due to historical reasons.
    # This special casing currently takes the form of:
    # - 'inputs' must be explicitly passed. A layer cannot have zero arguments,
    #   and inputs cannot have been provided via the default value of a kwarg.
    # - numpy/scalar values in `inputs` get converted to tensors
    # - implicit masks / mask metadata are only collected from 'inputs`
    # - Layers are built using shape info from 'inputs' only
    # - input_spec compatibility is only checked against `inputs`
    # - mixed precision casting (autocast) is only applied to `inputs`,
    #   not to any other argument.
    # - setting the SavedModel saving spec.
    inputs, args, kwargs = self._split_out_first_arg(args, kwargs)
    input_list = nest.flatten(inputs)

    # Functional Model construction mode is invoked when `Layer`s are called on
    # symbolic `KerasTensor`s, i.e.:
    # >> inputs = tf.keras.Input(10)
    # >> outputs = MyLayer()(inputs)  # Functional construction mode.
    # >> model = tf.keras.Model(inputs, outputs)
    if _in_functional_construction_mode(self, inputs, args, kwargs, input_list):
      return self._functional_construction_call(inputs, args, kwargs,
                                                input_list)

    # Maintains info about the `Layer.call` stack.
    call_context = base_layer_utils.call_context()

    # Accept NumPy and scalar inputs by converting to Tensors.
    if any(isinstance(x, (
        np_arrays.ndarray, np.ndarray, float, int)) for x in input_list):
      inputs = nest.map_structure(_convert_numpy_or_python_types, inputs)
      input_list = nest.flatten(inputs)

    # Handle `mask` propagation from previous layer to current layer. Masks can
    # be propagated explicitly via the `mask` argument, or implicitly via
    # setting the `_keras_mask` attribute on the inputs to a Layer. Masks passed
    # explicitly take priority.
    input_masks, mask_is_implicit = self._get_input_masks(
        inputs, input_list, args, kwargs)
    if self._expects_mask_arg and mask_is_implicit:
      kwargs['mask'] = input_masks

    # Training mode for `Layer.call` is set via (in order of priority):
    # (1) The `training` argument passed to this `Layer.call`, if it is not None
    # (2) The training mode of an outer `Layer.call`.
    # (3) The default mode set by `tf.keras.backend.set_learning_phase` (if set)
    # (4) Any non-None default value for `training` specified in the call
    #  signature
    # (5) False (treating the layer as if it's in inference)
    args, kwargs, training_mode = self._set_training_mode(
        args, kwargs, call_context)

    # Losses are cleared for all sublayers on the outermost `Layer.call`.
    # Losses are not cleared on inner `Layer.call`s, because sublayers can be
    # called multiple times.
    if not call_context.in_call:
      self._clear_losses()

    eager = context.executing_eagerly()
    with call_context.enter(
        layer=self,
        inputs=inputs,
        build_graph=not eager,
        training=training_mode):

      input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
      if eager:
        call_fn = self.call
        name_scope = self._name
      else:
        name_scope = self._name_scope()  # Avoid autoincrementing.  # pylint: disable=not-callable
        call_fn = self._autographed_call()

      with ops.name_scope_v2(name_scope):
        if not self.built:
          self._maybe_build(inputs)

        if self._autocast:
          inputs = self._maybe_cast_inputs(inputs, input_list)

        with autocast_variable.enable_auto_cast_variables(
            self._compute_dtype_object):
          outputs = call_fn(inputs, *args, **kwargs)

        if self._activity_regularizer:
          self._handle_activity_regularization(inputs, outputs)
        if self._supports_masking:
          self._set_mask_metadata(inputs, outputs, input_masks, not eager)
        if self._saved_model_inputs_spec is None:
          self._set_save_spec(inputs)

        return outputs

  def _functional_construction_call(self, inputs, args, kwargs, input_list):
    # [FUNCTION DOC]
    # HELPER: Internal path of the call wrapper used when the layer is applied to symbolic keras
    # tensors while constructing a functional model.
    # Inputs: inputs (structure of symbolic tensors), args and kwargs (extra call arguments) and
    # input_list (flattened inputs). Converts numpy/scalar inputs to tensors, resolves masks and the
    # training value (explicit, parent call context, global learning phase, call default),
    # adds framework-provided training/mask arguments temporarily, enters the call context in graph
    # building mode, produces symbolic outputs via the symbolic call helper, raises a value error
    # if the outputs are nothing (call returned None), removes the framework-provided arguments
    # again so they are not recorded in the graph node, and records connectivity metadata (an
    # inbound node). Returns the symbolic outputs.
    call_context = base_layer_utils.call_context()

    # Accept NumPy and scalar inputs by converting to Tensors.
    if any(isinstance(x, (
        np_arrays.ndarray, np.ndarray, float, int)) for x in input_list):

      def _convert_non_tensor(x):
        # [FUNCTION DOC]
        # HELPER: Mapping callback in the functional construction call.
        # Converts numpy arrays, Python floats and ints to tensors; returns any other value
        # (such as sparse tensors, which cannot be converted) unchanged.
        # Don't call `ops.convert_to_tensor` on all `inputs` because
        # `SparseTensors` can't be converted to `Tensor`.
        if isinstance(x, (np_arrays.ndarray, np.ndarray, float, int)):
          return tensor_conversion.convert_to_tensor_v2_with_dispatch(x)
        return x

      inputs = nest.map_structure(_convert_non_tensor, inputs)
      input_list = nest.flatten(inputs)

    # Handle `mask` propagation from previous layer to current layer. Masks can
    # be propagated explicitly via the `mask` argument, or implicitly via
    # setting the `_keras_mask` attribute on the inputs to a Layer. Masks passed
    # explicitly take priority.
    mask_arg_passed_by_framework = False
    input_masks, mask_is_implicit = self._get_input_masks(
        inputs, input_list, args, kwargs)
    if self._expects_mask_arg and mask_is_implicit:
      kwargs['mask'] = input_masks
      mask_arg_passed_by_framework = True

    # If `training` argument is None or not explicitly passed,
    # propagate `training` value from this layer's calling layer.
    training_value = None
    training_arg_passed_by_framework = False
    # Priority 1: `training` was explicitly passed a non-None value.
    if self._call_arg_was_passed('training', args, kwargs):
      training_value = self._get_call_arg_value('training', args, kwargs)
      if not self._expects_training_arg:
        kwargs.pop('training')

    if training_value is None:
      # Priority 2: `training` was passed to a parent layer.
      if call_context.training is not None:
        training_value = call_context.training
      # Priority 3: `learning_phase()` has been set.
      elif backend.global_learning_phase_is_set():
        training_value = backend.learning_phase()
        # Force the training_value to be bool type which matches to the contract
        # for layer/model call args.
        if tensor_util.is_tf_type(training_value):
          training_value = math_ops.cast(training_value, dtypes.bool)
        else:
          training_value = bool(training_value)
      # Priority 4: trace layer with the default training argument specified
      # in the `call` signature (or in inference mode if the `call` signature
      # specifies no non-None default).
      else:
        training_value = self._default_training_arg
      # In cases (2), (3), (4) the training argument is passed automatically
      # by the framework, and will not be hard-coded into the model.
      if self._expects_training_arg:
        args, kwargs = self._set_call_arg_value('training', training_value,
                                                args, kwargs)
        training_arg_passed_by_framework = True

    with call_context.enter(
        layer=self, inputs=inputs, build_graph=True, training=training_value):
      # Check input assumptions set after layer building, e.g. input shape.
      outputs = self._keras_tensor_symbolic_call(
          inputs, input_masks, args, kwargs)

      if outputs is None:
        raise ValueError('A layer\'s `call` method should return a '
                         'Tensor or a list of Tensors, not None '
                         '(layer: ' + self.name + ').')
      if training_arg_passed_by_framework:
        args, kwargs = self._set_call_arg_value(
            'training', None, args, kwargs, pop_kwarg_if_none=True)
      if mask_arg_passed_by_framework:
        kwargs.pop('mask')
      # Node connectivity does not special-case the first argument.
      outputs = self._set_connectivity_metadata((inputs,) + args, kwargs,
                                                outputs)
      return outputs

  def _set_training_mode(self, args, kwargs, call_context):
    # [FUNCTION DOC]
    # HELPER: Part of the call wrapper that resolves the training mode.
    # Inputs: args and kwargs (call arguments after the first), call_context (current call
    # stack info). If the call method takes a training argument, the value is taken from the
    # explicit argument, else the outer call context, else the global learning phase (cast to a
    # boolean or boolean tensor), else the default in the call signature (or inference), and
    # is written back into the arguments. If call has no training argument, a passed training
    # keyword is removed and becomes the mode for inner calls, otherwise the outer context's value
    # is used. Returns a tuple of updated args, updated kwargs and the resolved training mode.
    training_mode = None
    if self._expects_training_arg:
      # (1) `training` was passed to this `Layer.call`.
      if self._call_arg_was_passed('training', args, kwargs):
        training_mode = self._get_call_arg_value('training', args, kwargs)
      # If no `training` arg was passed, or `None` was explicitly passed,
      # the framework will make a decision about the training mode is.
      if training_mode is None:
        call_ctx_training = call_context.training
        # (2) `training` mode is inferred from an outer `Layer.call`.
        if call_ctx_training is not None:
          training_mode = call_ctx_training
        # (3) User set `tf.keras.backend.set_learning_phase`.
        elif backend.global_learning_phase_is_set():
          training_mode = backend.learning_phase()
          # Ensure value is a `bool` or `tf.bool`.
          if isinstance(training_mode, bool):
            pass
          elif tensor_util.is_tf_type(training_mode):
            training_mode = math_ops.cast(training_mode, dtypes.bool)
          else:
            training_mode = bool(training_mode)
        # (4) We default to using `call`'s default value for `training`,
        # or treating the layer as if it is in inference if no non-None default
        # is specified in the `call` signature.
        else:
          training_mode = self._default_training_arg

        # For case (2), (3), (4) `training` arg is passed by framework.
        args, kwargs = self._set_call_arg_value('training', training_mode, args,
                                                kwargs)
    else:
      if 'training' in kwargs:
        # `training` was passed to this `Layer` but is not needed for
        # `Layer.call`. It will set the default mode for inner `Layer.call`s.
        training_mode = kwargs.pop('training')
      else:
        # Grab the current `training` mode from any outer `Layer.call`.
        training_mode = call_context.training

    return args, kwargs, training_mode

  def _autographed_call(self):
    # [FUNCTION DOC]
    # HELPER: Part of the call wrapper (graph mode) choosing the callable to execute.
    # Returns the autograph-converted call method for subclassed layers that were not loaded from
    # a saved model (respecting an enclosing autograph setting); otherwise returns call as is.
    # Takes no arguments, no side effects.
    # Wrapping `call` function in autograph to allow for dynamic control
    # flow and control dependencies in call. We are limiting this to
    # subclassed layers as autograph is strictly needed only for
    # subclassed layers and models.
    # tf_convert will respect the value of autograph setting in the
    # enclosing tf.function, if any.
    if (base_layer_utils.is_subclassed(self) and
        not base_layer_utils.from_saved_model(self)):
      return autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
    else:
      return self.call

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property: the dtype of the layer's variables, i.e. the variable dtype of the dtype
    # policy (may be nothing if the policy is still to be inferred).
    """The dtype of the layer weights.

    This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
    mixed precision is used, this is the same as `Layer.compute_dtype`, the
    dtype of the layer's computations.
    """
    return self._dtype_policy.variable_dtype

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the layer's string name chosen in the constructor.
    """Name of the layer (string), set in the constructor."""
    return self._name

  @property
  def supports_masking(self):
    # [FUNCTION DOC]
    # Property returning whether the layer computes and propagates masks. True by default only if
    # compute_mask is overridden, otherwise whatever a subclass set.
    """Whether this layer supports computing a mask using `compute_mask`."""
    return self._supports_masking

  @supports_masking.setter
  def supports_masking(self, value):
    # [FUNCTION DOC]
    # Setter for the masking-support flag. Input: value, a boolean. Stores it; returns nothing.
    self._supports_masking = value

  @property
  def dynamic(self):
    # [FUNCTION DOC]
    # Read-only property: true if this layer or any nested sub-layer was created as dynamic
    # (eager-only), computed by scanning the layer and all its flattened sub-layers.
    """Whether the layer is dynamic (eager-only); set in the constructor."""
    return any(layer._dynamic for layer in self._flatten_layers())

  @property
  @doc_controls.do_not_doc_inheritable
  def stateful(self):
    # [FUNCTION DOC]
    # Read-only property: true if this layer or any nested sub-layer is stateful (its updates run
    # during inference as well, as for stateful recurrent layers).
    return any(layer._stateful for layer in self._flatten_layers())

  @stateful.setter
  def stateful(self, value):
    # [FUNCTION DOC]
    # Setter for the stateful flag of this layer only (not sub-layers). Input: value, a boolean.
    self._stateful = value

  @property
  def trainable(self):
    # [FUNCTION DOC]
    # Property returning this layer's own trainable flag (sub-layers are not consulted).
    return self._trainable

  @trainable.setter
  def trainable(self, value):
    # [FUNCTION DOC]
    # Setter that applies the given boolean trainable value to this layer and, recursively, to all
    # nested sub-layers. Returns nothing.
    for layer in self._flatten_layers():
      layer._trainable = value

  @property
  def activity_regularizer(self):
    # [FUNCTION DOC]
    # Property returning the optional regularizer applied to this layer's outputs (or nothing).
    """Optional regularizer function for the output of this layer."""
    return self._activity_regularizer

  @activity_regularizer.setter
  def activity_regularizer(self, regularizer):
    # [FUNCTION DOC]
    # Setter storing the given regularizer callable (or nothing) as the activity regularizer;
    # no resolution from string identifiers or validation is performed here.
    """Optional regularizer function for the output of this layer."""
    self._activity_regularizer = regularizer

  @property
  def input_spec(self):
    # [FUNCTION DOC]
    # Property returning the input specification (a single input spec, nested structure of them,
    # or nothing) used to validate inputs when the layer is called.
    """`InputSpec` instance(s) describing the input format for this layer.

    When you create a layer subclass, you can set `self.input_spec` to enable
    the layer to run input compatibility checks when it is called.
    Consider a `Conv2D` layer: it can only be called on a single input tensor
    of rank 4. As such, you can set, in `__init__()`:

    ```python
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    ```

    Now, if you try to call the layer on an input that isn't rank 4
    (for instance, an input of shape `(2,)`, it will raise a nicely-formatted
    error:

    ```
    ValueError: Input 0 of layer conv2d is incompatible with the layer:
    expected ndim=4, found ndim=1. Full shape received: [2]
    ```

    Input checks that can be specified via `input_spec` include:
    - Structure (e.g. a single input, a list of 2 inputs, etc)
    - Shape
    - Rank (ndim)
    - Dtype

    For more information, see `tf.keras.layers.InputSpec`.

    Returns:
      A `tf.keras.layers.InputSpec` instance, or nested structure thereof.
    """
    return self._input_spec

  @input_spec.setter
  # Must be decorated to prevent tracking, since the input_spec can be nested
  # InputSpec objects.
  @trackable.no_automatic_dependency_tracking
  def input_spec(self, value):
    # [FUNCTION DOC]
    # Setter for the input specification (not tracked automatically). Input: value, an input spec,
    # nested structure of input specs, or nothing. Raises a type error if any non-empty leaf is not
    # an input spec instance. Stores the value; returns nothing.
    for v in nest.flatten(value):
      if v is not None and not isinstance(v, InputSpec):
        raise TypeError('Layer input_spec must be an instance of InputSpec. '
                        'Got: {}'.format(v))
    self._input_spec = value

  @property
  def trainable_weights(self):
    # [FUNCTION DOC]
    # Read-only property: list of trainable variables of this layer plus its sub-layers, with
    # duplicates removed. Returns an empty list when the layer's trainable flag is false.
    """List of all trainable weights tracked by this layer.

    Trainable weights are updated via gradient descent during training.

    Returns:
      A list of trainable variables.
    """
    if self.trainable:
      children_weights = self._gather_children_attribute('trainable_variables')
      return self._dedup_weights(self._trainable_weights + children_weights)
    else:
      return []

  @property
  def non_trainable_weights(self):
    # [FUNCTION DOC]
    # Read-only property: list of non-trainable variables of this layer plus its sub-layers, with
    # duplicates removed. If the layer's trainable flag is false, all its weights (including the
    # normally trainable ones, and all sub-layer variables) are reported here instead.
    """List of all non-trainable weights tracked by this layer.

    Non-trainable weights are *not* updated during training. They are expected
    to be updated manually in `call()`.

    Returns:
      A list of non-trainable variables.
    """
    if self.trainable:
      children_weights = self._gather_children_attribute(
          'non_trainable_variables')
      non_trainable_weights = self._non_trainable_weights + children_weights
    else:
      children_weights = self._gather_children_attribute('variables')
      non_trainable_weights = (
          self._trainable_weights + self._non_trainable_weights +
          children_weights)
    return self._dedup_weights(non_trainable_weights)

  @property
  def weights(self):
    # [FUNCTION DOC]
    # Read-only property: list of all weights, the trainable weights followed by the
    # non-trainable weights, each including those of nested sub-layers.
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.trainable_weights + self.non_trainable_weights

  @property
  @doc_controls.do_not_generate_docs
  def updates(self):
    # [FUNCTION DOC]
    # Deprecated read-only property kept for backward compatibility. Emits a warning that it will be
    # removed and always returns an empty list, since updates are applied automatically in
    # TensorFlow 2.
    warnings.warn('`layer.updates` will be removed in a future version. '
                  'This property should not be used in TensorFlow 2.0, '
                  'as `updates` are applied automatically.')
    return []

  @property
  def losses(self):
    # [FUNCTION DOC]
    # Read-only property collecting losses from this layer and all nested sub-layers. For each
    # layer it uses the eager losses of the current call if any exist (skipping the placeholder
    # loss that revived layers may carry), otherwise the symbolic losses; then it evaluates every
    # zero-argument loss callable (such as weight regularizers) and keeps non-empty results.
    # Returns a list of loss tensors. Evaluating callables happens on each access.
    """List of losses added using the `add_loss()` API.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Examples:

    >>> class MyLayer(tf.keras.layers.Layer):
    ...   def call(self, inputs):
    ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    ...     return inputs
    >>> l = MyLayer()
    >>> l(np.ones((10, 1)))
    >>> l.losses
    [1.0]

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Activity regularization.
    >>> len(model.losses)
    0
    >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
    >>> len(model.losses)
    1

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
    >>> x = d(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Weight regularization.
    >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
    >>> model.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

    Returns:
      A list of tensors.
    """
    collected_losses = []
    for layer in self._flatten_layers():
      # If any eager losses are present, we assume the model to be part of an
      # eager training loop (either a custom one or the one used when
      # `run_eagerly=True`) and so we always return just the eager losses.
      if layer._eager_losses:
        # Filter placeholder losses that may have been added by revived layers.
        # (see base_layer_utils for details).
        if (layer._eager_losses[0] is
            not base_layer_utils.REVIVED_LOSS_PLACEHOLDER):
          collected_losses.extend(layer._eager_losses)
      else:
        collected_losses.extend(layer._losses)
      for regularizer in layer._callable_losses:
        loss_tensor = regularizer()
        if loss_tensor is not None:
          collected_losses.append(loss_tensor)
    return collected_losses

  def add_loss(self, losses, **kwargs):
    # [FUNCTION DOC]
    # Registers one or more losses on the layer. Input: losses, a tensor, number, nothing, a
    # zero-argument callable returning a loss, or a nested structure of these (flattened). Keyword
    # arguments: an inputs entry is accepted and ignored (deprecated); any other keyword raises a
    # type error. Callables are stored (wrapped so that they run without autocasting and are
    # tagged as unconditional losses) and evaluated lazily; non-tensor values are converted to
    # tensors of the backend float type; symbolic tensors (outside tf functions) are stored as
    # symbolic losses (or recorded on the graph network for functional models); other tensors are
    # eager losses. Raises a value error if eager tensors are added outside of a layer call.
    # Returns nothing; mutates the layer's loss lists.
    """Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This method can be used inside a subclassed layer or model's `call`
    function, in which case `losses` should be a Tensor or list of Tensors.

    Example:

    ```python
    class MyLayer(tf.keras.layers.Layer):
      def call(self, inputs):
        self.add_loss(tf.abs(tf.reduce_mean(inputs)))
        return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any loss Tensors passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    losses become part of the model's topology and are tracked in `get_config`.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Activity regularization.
    model.add_loss(tf.abs(tf.reduce_mean(x)))
    ```

    If this is not the case for your loss (if, for example, your loss references
    a `Variable` of one of the model's layers), you can wrap your loss in a
    zero-argument lambda. These losses are not tracked as part of the model's
    topology since they can't be serialized.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    d = tf.keras.layers.Dense(10)
    x = d(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Weight regularization.
    model.add_loss(lambda: tf.reduce_mean(d.kernel))
    ```

    Args:
      losses: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
        may also be zero-argument callables which create a loss tensor.
      **kwargs: Additional keyword arguments for backward compatibility.
        Accepted values:
          inputs - Deprecated, will be automatically inferred.
    """
    kwargs.pop('inputs', None)
    if kwargs:
      raise TypeError('Unknown keyword arguments: %s' % (kwargs.keys(),))

    def _tag_callable(loss):
      # [FUNCTION DOC]
      # HELPER: Wrapper used by add_loss for callable losses.
      # Input: loss, a zero-argument callable, a tensor, a number or nothing. Callables are invoked
      # with auto-casting disabled. Returns nothing for an empty result, otherwise a tensor
      # (numbers converted using the backend float type) with an unconditional-loss marker
      # attribute set to true.
      """Tags callable loss tensor as `_unconditional_loss`."""
      if callable(loss):
        # We run the loss without autocasting, as regularizers are often
        # numerically unstable in float16.
        with autocast_variable.enable_auto_cast_variables(None):
          loss = loss()
      if loss is None:
        return None  # Will be filtered out when computing the .losses property
      if not tensor_util.is_tf_type(loss):
        loss = tensor_conversion.convert_to_tensor_v2_with_dispatch(
            loss, dtype=backend.floatx()
        )
      loss._unconditional_loss = True  # pylint: disable=protected-access
      return loss

    losses = nest.flatten(losses)

    callable_losses = []
    eager_losses = []
    symbolic_losses = []
    for loss in losses:
      if callable(loss):
        callable_losses.append(functools.partial(_tag_callable, loss))
        continue
      if loss is None:
        continue
      if not tensor_util.is_tf_type(loss) and not isinstance(
          loss, keras_tensor.KerasTensor):
        loss = tensor_conversion.convert_to_tensor_v2_with_dispatch(
            loss, dtype=backend.floatx()
        )
      # TF Functions should take the eager path.
      if ((tf_utils.is_symbolic_tensor(loss) or
           isinstance(loss, keras_tensor.KerasTensor)) and
          not base_layer_utils.is_in_tf_function()):
        symbolic_losses.append(loss)
      elif tensor_util.is_tf_type(loss):
        eager_losses.append(loss)

    self._callable_losses.extend(callable_losses)

    in_call_context = base_layer_utils.call_context().in_call
    if eager_losses and not in_call_context:
      raise ValueError(
          'Expected a symbolic Tensors or a callable for the loss value. '
          'Please wrap your loss computation in a zero argument `lambda`.')

    self._eager_losses.extend(eager_losses)

    for symbolic_loss in symbolic_losses:
      if getattr(self, '_is_graph_network', False):
        self._graph_network_add_loss(symbolic_loss)
      else:
        # Possible a loss was added in a Layer's `build`.
        self._losses.append(symbolic_loss)

  def _clear_losses(self):
    # [FUNCTION DOC]
    # HELPER: Called at the start of each outermost layer call.
    # Resets the thread-local eager loss list to empty, for this layer only when it has no tracked
    # sub-objects, otherwise for this layer and every nested sub-layer. Returns nothing.
    """Used every step in eager to reset losses."""
    # Set to thread local directly to avoid Layer.__setattr__ overhead.
    if not getattr(self, '_self_tracked_trackables',
                   None):  # Fast path for single Layer.
      self._thread_local._eager_losses = []
    else:
      for layer in self._flatten_layers():
        layer._thread_local._eager_losses = []

  @property
  def metrics(self):
    # [FUNCTION DOC]
    # Read-only property: list of metric objects added through add_metric on this layer and all
    # nested sub-layers, collected under each layer's metrics lock.
    """List of metrics added using the `add_metric()` API.

    Example:

    >>> input = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2)
    >>> output = d(input)
    >>> d.add_metric(tf.reduce_max(output), name='max')
    >>> d.add_metric(tf.reduce_min(output), name='min')
    >>> [m.name for m in d.metrics]
    ['max', 'min']

    Returns:
      A list of `Metric` objects.
    """
    collected_metrics = []
    for layer in self._flatten_layers():
      with layer._metrics_lock:
        collected_metrics.extend(layer._metrics)
    return collected_metrics

  def add_metric(self, value, name=None, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says an aggregation keyword controls aggregation, but the value is
    # only validated and never used: non-metric values are always averaged with a mean metric.
    # Also, the final branch computes the aggregation as nothing when the value came from a metric
    # object, but that case has already raised an error just before, so that branch is dead.
    # Adds a metric to the layer. Inputs: value, a metric result tensor or a symbolic tensor;
    # name, a string required unless the value came from a metric object; keyword arguments may
    # only contain an aggregation entry, otherwise a type error is raised. Raises a value error
    # if no name is given for a plain tensor, if a non-symbolic value is added outside a call, or
    # if a metric-object result is added to a functional model outside a call. Inside a call (or
    # for non-graph-network layers) it finds an existing metric of that name or creates a mean
    # metric (or adopts the metric object), appends it to the metrics list under the lock and
    # updates its state with the value. For functional models it records the metric on the graph
    # network. Returns nothing.
    """Adds metric tensor to the layer.

    This method can be used inside the `call()` method of a subclassed layer
    or model.

    ```python
    class MyMetricLayer(tf.keras.layers.Layer):
      def __init__(self):
        super(MyMetricLayer, self).__init__(name='my_metric_layer')
        self.mean = tf.keras.metrics.Mean(name='metric_1')

      def call(self, inputs):
        self.add_metric(self.mean(inputs))
        self.add_metric(tf.reduce_sum(inputs), name='metric_2')
        return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any tensor passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    metrics become part of the model's topology and are tracked when you
    save the model via `save()`.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(math_ops.reduce_sum(x), name='metric_1')
    ```

    Note: Calling `add_metric()` with the result of a metric object on a
    Functional Model, as shown in the example below, is not supported. This is
    because we cannot trace the metric result tensor back to the model's inputs.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
    ```

    Args:
      value: Metric tensor.
      name: String metric name.
      **kwargs: Additional keyword arguments for backward compatibility.
        Accepted values:
        `aggregation` - When the `value` tensor provided is not the result of
        calling a `keras.Metric` instance, it will be aggregated by default
        using a `keras.Metric.Mean`.
    """
    kwargs_keys = list(kwargs.keys())
    if (len(kwargs_keys) > 1 or
        (len(kwargs_keys) == 1 and kwargs_keys[0] != 'aggregation')):
      raise TypeError('Unknown keyword arguments: ', str(kwargs.keys()))

    from_metric_obj = hasattr(value, '_metric_obj')
    is_symbolic = isinstance(value, keras_tensor.KerasTensor)
    in_call_context = base_layer_utils.call_context().in_call

    if name is None and not from_metric_obj:
      # Eg. `self.add_metric(math_ops.reduce_sum(x))`
      # In eager mode, we use metric name to lookup a metric. Without a name,
      # a new Mean metric wrapper will be created on every model/layer call.
      # So, we raise an error when no name is provided.
      # We will do the same for symbolic mode for consistency although a name
      # will be generated if no name is provided.

      # We will not raise this error in the foll use case for the sake of
      # consistency as name in provided in the metric constructor.
      # mean = metrics.Mean(name='my_metric')
      # model.add_metric(mean(outputs))
      raise ValueError('Please provide a name for your metric like '
                       '`self.add_metric(tf.reduce_sum(inputs), '
                       'name=\'mean_activation\')`')
    elif from_metric_obj:
      name = value._metric_obj.name

    if not in_call_context and not is_symbolic:
      raise ValueError('Expected a symbolic Tensor for the metric value, '
                       'received: ' + str(value))

    # If a metric was added in a Layer's `call` or `build`.
    if in_call_context or not getattr(self, '_is_graph_network', False):
      # TF Function path should take the eager path.

      # If the given metric is available in `metrics` list we just update state
      # on it, otherwise we create a new metric instance and
      # add it to the `metrics` list.
      metric_obj = getattr(value, '_metric_obj', None)
      # Tensors that come from a Metric object already updated the Metric state.
      should_update_state = not metric_obj
      name = metric_obj.name if metric_obj else name

      with self._metrics_lock:
        match = self._get_existing_metric(name)
        if match:
          metric_obj = match
        elif metric_obj:
          self._metrics.append(metric_obj)
        else:
          # Build the metric object with the value's dtype if it defines one
          metric_obj = metrics_mod.Mean(
              name=name, dtype=getattr(value, 'dtype', None))
          self._metrics.append(metric_obj)

      if should_update_state:
        metric_obj(value)
    else:
      if from_metric_obj:
        raise ValueError('Using the result of calling a `Metric` object '
                         'when calling `add_metric` on a Functional '
                         'Model is not supported. Please pass the '
                         'Tensor to monitor directly.')

      # Insert layers into the Keras Graph Network.
      aggregation = None if from_metric_obj else 'mean'
      self._graph_network_add_metric(value, aggregation, name)

  @doc_controls.do_not_doc_inheritable
  def add_update(self, updates, inputs=None):
    # [FUNCTION DOC]
    # Runs layer update callables. Inputs: updates, an update op, callable or nested structure of
    # them; inputs, deprecated (a warning is logged if given). Does nothing during functional model
    # construction or when the call context is frozen (layer not trainable). Otherwise it calls
    # each callable in the flattened updates; non-callable entries are ignored because in eager
    # or TF2 execution updates already ran. Returns nothing.
    """Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This call is ignored when eager execution is enabled (in that case, variable
    updates are run on the fly and thus do not need to be tracked for later
    execution).

    Args:
      updates: Update op, or list/tuple of update ops, or zero-arg callable
        that returns an update op. A zero-arg callable should be passed in
        order to disable running the updates by setting `trainable=False`
        on this Layer, when executing in Eager mode.
      inputs: Deprecated, will be automatically inferred.
    """
    if inputs is not None:
      tf_logging.warning(
          '`add_update` `inputs` kwarg has been deprecated. You no longer need '
          'to pass a value to `inputs` as it is being automatically inferred.')
    call_context = base_layer_utils.call_context()
    # No need to run updates during Functional API construction.
    if call_context.in_keras_graph:
      return

    # Callable updates are disabled by setting `trainable=False`.
    if not call_context.frozen:
      for update in nest.flatten(updates):
        if callable(update):
          update()  # pylint: disable=not-callable

  def set_weights(self, weights):
    # [FUNCTION DOC]
    # Assigns weight values from a list of NumPy arrays, in the order of the layer's weights
    # property. Raises a value error if the number of arrays differs from the expected count (a
    # handler for a trackable object may consume several arrays) or if any array shape is not
    # compatible with its variable. Side effects: trackable weight handlers are set as they are
    # visited (before later shape errors can be raised), variables are assigned in one batch, and
    # finalize_state is then called on the layer and all nested sub-layers. Returns nothing.
    """Sets the weights of the layer, from NumPy arrays.

    The weights of a layer represent the state of the layer. This function
    sets the weight values from numpy arrays. The weight values should be
    passed in the order they are created by the layer. Note that the layer's
    weights must be instantiated before calling this function, by calling
    the layer.

    For example, a `Dense` layer returns a list of two values: the kernel matrix
    and the bias vector. These can be used to set the weights of another
    `Dense` layer:

    >>> layer_a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> layer_a.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> layer_b.get_weights()
    [array([[2.],
           [2.],
           [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b.set_weights(layer_a.get_weights())
    >>> layer_b.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]

    Args:
      weights: a list of NumPy arrays. The number
        of arrays and their shape must match
        number of the dimensions of the weights
        of the layer (i.e. it should match the
        output of `get_weights`).

    Raises:
      ValueError: If the provided weights list does not match the
        layer's specifications.
    """
    params = self.weights

    expected_num_weights = 0
    for param in params:
      if isinstance(param, base_layer_utils.TrackableWeightHandler):
        expected_num_weights += param.num_tensors
      else:
        expected_num_weights += 1

    if expected_num_weights != len(weights):
      raise ValueError(
          'You called `set_weights(weights)` on layer "%s" '
          'with a weight list of length %s, but the layer was '
          'expecting %s weights. Provided weights: %s...' %
          (self.name, len(weights), expected_num_weights, str(weights)[:50]))

    weight_index = 0
    weight_value_tuples = []
    for param in params:
      if isinstance(param, base_layer_utils.TrackableWeightHandler):
        num_tensors = param.num_tensors
        tensors = weights[weight_index:weight_index + num_tensors]
        param.set_weights(tensors)
        weight_index += num_tensors
      else:
        weight = weights[weight_index]
        weight_shape = weight.shape if hasattr(weight, 'shape') else ()
        ref_shape = param.shape
        if not ref_shape.is_compatible_with(weight_shape):
          raise ValueError(
              'Layer weight shape %s not compatible with provided weight '
              'shape %s' % (ref_shape, weight_shape))
        weight_value_tuples.append((param, weight))
        weight_index += 1

    backend.batch_set_value(weight_value_tuples)

    # Perform any layer defined finalization of the layer state.
    for layer in self._flatten_layers():
      layer.finalize_state()

  def get_weights(self):
    # [FUNCTION DOC]
    # Returns the current values of all weights (trainable first, then non-trainable) as a list of
    # NumPy arrays; handlers for trackable objects contribute all their tensors. Takes no arguments.
    """Returns the current weights of the layer, as NumPy arrays.

    The weights of a layer represent the state of the layer. This function
    returns both trainable and non-trainable weight values associated with this
    layer as a list of NumPy arrays, which can in turn be used to load state
    into similarly parameterized layers.

    For example, a `Dense` layer returns a list of two values: the kernel matrix
    and the bias vector. These can be used to set the weights of another
    `Dense` layer:

    >>> layer_a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> layer_a.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> layer_b.get_weights()
    [array([[2.],
           [2.],
           [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b.set_weights(layer_a.get_weights())
    >>> layer_b.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]

    Returns:
        Weights values as a list of NumPy arrays.
    """
    weights = self.weights
    output_weights = []
    for weight in weights:
      if isinstance(weight, base_layer_utils.TrackableWeightHandler):
        output_weights.extend(weight.get_tensors())
      else:
        output_weights.append(weight)
    return backend.batch_get_value(output_weights)

  @doc_controls.do_not_generate_docs
  def finalize_state(self):
    # [FUNCTION DOC]
    # Hook called after weights are set; the default does nothing. Subclasses may override it to
    # refresh derived state. Takes no arguments and returns nothing.
    """Finalizes the layers state after updating layer weights.

    This function can be subclassed in a layer and will be called after updating
    a layer weights. It can be overridden to finalize any additional layer state
    after a weight update.
    """
    pass

  @doc_controls.do_not_generate_docs
  def get_updates_for(self, inputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it returns updates that depend on the given inputs, but the
    # inputs are ignored and the result of the deprecated updates property (always empty) is
    # returned. Emits a deprecation warning. Returns an empty list.
    """Deprecated, do NOT use!

    Retrieves updates relevant to a specific set of inputs.

    Args:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of update ops of the layer that depend on `inputs`.
    """
    warnings.warn('`layer.get_updates_for` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.updates` method instead.')
    return self.updates

  @doc_controls.do_not_generate_docs
  def get_losses_for(self, inputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it returns losses relevant to the given inputs, but the
    # inputs are ignored and all losses of the layer (and sub-layers) are returned. Emits a
    # deprecation warning. Returns a list of loss tensors.
    """Deprecated, do NOT use!

    Retrieves losses relevant to a specific set of inputs.

    Args:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of loss tensors of the layer that depend on `inputs`.
    """
    warnings.warn('`layer.get_losses_for` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.losses` instead.')
    return self.losses

  @doc_controls.do_not_doc_inheritable
  def get_input_mask_at(self, node_index):
    # [FUNCTION DOC]
    # Returns the mask(s) attached to the input tensor(s) of the inbound node at the given integer
    # index: a list of masks (entries may be nothing) when the node has several inputs, otherwise
    # the single mask or nothing. Errors from looking up the node's inputs propagate.
    """Retrieves the input mask tensor(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple inputs).
    """
    inputs = self.get_input_at(node_index)
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  @doc_controls.do_not_doc_inheritable
  def get_output_mask_at(self, node_index):
    # [FUNCTION DOC]
    # Returns the mask(s) attached to the output tensor(s) of the node at the given integer index:
    # a list of masks when the node has several outputs, otherwise a single mask or nothing.
    # Errors from looking up the node's outputs propagate.
    """Retrieves the output mask tensor(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple outputs).
    """
    output = self.get_output_at(node_index)
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  @property
  @doc_controls.do_not_doc_inheritable
  def input_mask(self):
    # [FUNCTION DOC]
    # Read-only property: the mask(s) of the layer's single input (list when there are several
    # inputs, entries may be nothing). Errors from the input property (for example when the layer
    # has several inbound nodes) propagate.
    """Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input mask tensor (potentially None) or list of input
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    inputs = self.input
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  @property
  @doc_controls.do_not_doc_inheritable
  def output_mask(self):
    # [FUNCTION DOC]
    # Read-only property: the mask(s) of the layer's single output (list when there are several
    # outputs, entries may be nothing). Errors from the output property propagate.
    """Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Output mask tensor (potentially None) or list of output
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    output = self.output
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  @doc_controls.do_not_doc_inheritable
  def get_input_shape_at(self, node_index):
    # [FUNCTION DOC]
    # Returns the input shape(s) recorded on the inbound node at the given integer index, via the
    # node attribute lookup helper (which raises errors for bad indices or eager mode).
    """Retrieves the input shape(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    return self._get_node_attribute_at_index(node_index, 'input_shapes',
                                             'input shape')

  @doc_controls.do_not_doc_inheritable
  def get_output_shape_at(self, node_index):
    # [FUNCTION DOC]
    # Returns the output shape(s) recorded on the node at the given integer index, via the node
    # attribute lookup helper (which raises errors for bad indices or eager mode).
    """Retrieves the output shape(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    return self._get_node_attribute_at_index(node_index, 'output_shapes',
                                             'output shape')

  @doc_controls.do_not_doc_inheritable
  def get_input_at(self, node_index):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it raises a runtime error in eager mode; the code has no eager
    # check. A runtime error is raised only when the layer has never been called, and a value error
    # when the index is out of range.
    # Returns the input tensor(s) of the inbound node at the given integer index (a single tensor
    # if the node has exactly one input, otherwise a list), via the node attribute helper.
    """Retrieves the input tensor(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first input node of the layer.

    Returns:
        A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    return self._get_node_attribute_at_index(node_index, 'input_tensors',
                                             'input')

  @doc_controls.do_not_doc_inheritable
  def get_output_at(self, node_index):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it raises a runtime error in eager mode; the code has no eager
    # check. A runtime error is raised only when the layer has never been called, and a value error
    # when the index is out of range.
    # Returns the output tensor(s) of the inbound node at the given integer index (a single tensor
    # if the node has exactly one output, otherwise a list), via the node attribute helper.
    """Retrieves the output tensor(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first output node of the layer.

    Returns:
        A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    return self._get_node_attribute_at_index(node_index, 'output_tensors',
                                             'output')

  @property
  def input(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it is only applicable with exactly one input and raises a
    # runtime error in eager mode; the code never checks either and simply returns the inputs of
    # the first inbound node, even if the layer was called several times.
    # Read-only property returning the input tensor(s) of the first inbound node. Raises an
    # attribute error if the layer has no inbound nodes (was never connected).
    """Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    """
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name +
                           ' is not connected, no input to return.')
    return self._get_node_attribute_at_index(0, 'input_tensors', 'input')

  @property
  def output(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it raises an attribute error when connected to more than one
    # incoming layer and a runtime error in eager mode; the code checks neither and returns the
    # outputs of the first inbound node.
    # Read-only property returning the output tensor(s) of the first inbound node. Raises an
    # attribute error if the layer has no inbound nodes.
    """Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
      Output tensor or list of output tensors.

    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    """
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name + ' has no inbound nodes.')
    return self._get_node_attribute_at_index(0, 'output_tensors', 'output')

  @property
  @doc_controls.do_not_doc_inheritable
  def input_shape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring mentions a runtime error in eager mode, which the code never
    # raises; only attribute errors are raised.
    # Read-only property returning the input shape(s) shared by all inbound nodes (compared by their
    # text form). Raises an attribute error if the layer was never called or if inbound nodes have
    # different input shapes.
    """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined input shape.')
    all_input_shapes = set(
        [str(node.input_shapes) for node in self._inbound_nodes])
    if len(all_input_shapes) == 1:
      return self._inbound_nodes[0].input_shapes
    else:
      raise AttributeError('The layer "' + str(self.name) +
                           ' has multiple inbound nodes, '
                           'with different input shapes. Hence '
                           'the notion of "input shape" is '
                           'ill-defined for the layer. '
                           'Use `get_input_shape_at(node_index)` '
                           'instead.')

  def count_params(self):
    # [FUNCTION DOC]
    # Returns the total number of scalar values in all weights of the layer as an integer. If the
    # layer is not built: for a graph network (functional model) it builds it from its inputs
    # (side effect), otherwise it raises a value error telling the user to build it first.
    """Count the total number of scalars composing the weights.

    Returns:
        An integer count.

    Raises:
        ValueError: if the layer isn't yet built
          (in which case its weights aren't yet defined).
    """
    if not self.built:
      if getattr(self, '_is_graph_network', False):
        with tf_utils.maybe_init_scope(self):
          self._maybe_build(self.inputs)
      else:
        raise ValueError('You tried to call `count_params` on ' + self.name +
                         ', but the layer isn\'t built. '
                         'You can build it manually via: `' + self.name +
                         '.build(batch_input_shape)`.')
    return layer_utils.count_params(self.weights)

  @property
  @doc_controls.do_not_doc_inheritable
  def output_shape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring mentions a runtime error in eager mode, which the code never
    # raises; only attribute errors are raised.
    # Read-only property returning the output shape(s) shared by all inbound nodes (compared by
    # their text form). Raises an attribute error if the layer was never called or if inbound nodes
    # have different output shapes.
    """Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    """
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined output shape.')
    all_output_shapes = set(
        [str(node.output_shapes) for node in self._inbound_nodes])
    if len(all_output_shapes) == 1:
      return self._inbound_nodes[0].output_shapes
    else:
      raise AttributeError('The layer "%s"'
                           ' has multiple inbound nodes, '
                           'with different output shapes. Hence '
                           'the notion of "output shape" is '
                           'ill-defined for the layer. '
                           'Use `get_output_shape_at(node_index)` '
                           'instead.' % self.name)

  @property
  @doc_controls.do_not_doc_inheritable
  def inbound_nodes(self):
    # [FUNCTION DOC]
    # Deprecated read-only property exposing the list of nodes through which this layer was called,
    # kept for compatibility with external Keras.
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._inbound_nodes

  @property
  @doc_controls.do_not_doc_inheritable
  def outbound_nodes(self):
    # [FUNCTION DOC]
    # Deprecated read-only property exposing the list of nodes that consume this layer's outputs,
    # kept for compatibility with external Keras.
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._outbound_nodes

  ##############################################################################
  # Methods & attributes below are public aliases of other methods.            #
  ##############################################################################

  @doc_controls.do_not_doc_inheritable
  def apply(self, inputs, *args, **kwargs):
    # [FUNCTION DOC]
    # Deprecated alias of the call wrapper. Emits a deprecation warning, then applies the layer to
    # inputs with the extra positional and keyword arguments and returns the outputs.
    """Deprecated, do NOT use!

    This is an alias of `self.__call__`.

    Args:
      inputs: Input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).
    """
    warnings.warn('`layer.apply` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.__call__` method instead.')
    return self.__call__(inputs, *args, **kwargs)

  @doc_controls.do_not_doc_inheritable
  def add_variable(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Deprecated alias of add_weight. Emits a deprecation warning, forwards all arguments, and
    # returns the created variable.
    """Deprecated, do NOT use! Alias for `add_weight`."""
    warnings.warn('`layer.add_variable` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.add_weight` method instead.')
    return self.add_weight(*args, **kwargs)

  @property
  @doc_controls.do_not_generate_docs
  def variables(self):
    # [FUNCTION DOC]
    # Read-only property, alias of the weights property (trainable then non-trainable weights).
    """Returns the list of all layer variables/weights.

    Alias of `self.weights`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
      A list of variables.
    """
    return self.weights

  @property
  @doc_controls.do_not_generate_docs
  def trainable_variables(self):
    # [FUNCTION DOC]
    # Read-only property, alias of the trainable weights property.
    return self.trainable_weights

  @property
  @doc_controls.do_not_generate_docs
  def non_trainable_variables(self):
    # [FUNCTION DOC]
    # Read-only property, alias of the non-trainable weights property.
    return self.non_trainable_weights

  ##############################################################################
  # Methods & attributes below are all private and only used by the framework. #
  ##############################################################################

  @property
  def _inbound_nodes(self):
    # [FUNCTION DOC]
    # HELPER: Internal graph-connectivity bookkeeping.
    # Property returning the list of nodes created each time this layer was called symbolically.
    return self._inbound_nodes_value

  @_inbound_nodes.setter
  @trackable.no_automatic_dependency_tracking
  def _inbound_nodes(self, value):
    # [FUNCTION DOC]
    # HELPER: Internal graph-connectivity bookkeeping.
    # Setter storing the given list of inbound nodes without automatic dependency tracking.
    self._inbound_nodes_value = value

  @property
  def _outbound_nodes(self):
    # [FUNCTION DOC]
    # HELPER: Internal graph-connectivity bookkeeping.
    # Property returning the list of nodes that consume this layer's outputs.
    return self._outbound_nodes_value

  @_outbound_nodes.setter
  @trackable.no_automatic_dependency_tracking
  def _outbound_nodes(self, value):
    # [FUNCTION DOC]
    # HELPER: Internal graph-connectivity bookkeeping.
    # Setter storing the given list of outbound nodes without automatic dependency tracking.
    self._outbound_nodes_value = value

  def _set_dtype_policy(self, dtype):
    # [FUNCTION DOC]
    # HELPER: Used by the constructor, variable creation and input-based dtype inference.
    # Sets the layer's dtype policy and cached compute dtype object. Input: dtype, which may be a
    # policy object, a serialized policy mapping, a mixed-precision policy name, any dtype-like
    # value (converted to a policy named after it), or empty (use the global policy). Raises a value
    # error if the mixed float16 policy is used with a distribution strategy that does not support
    # loss scaling. Returns nothing.
    """Sets self._dtype_policy."""
    if isinstance(dtype, policy.Policy):
      self._dtype_policy = dtype
    elif isinstance(dtype, dict):
      self._dtype_policy = policy.deserialize(dtype)
    elif isinstance(dtype, str) and dtype in ('mixed_float16',
                                              'mixed_bfloat16'):
      # The isinstance check is required since np.dtype raises an error if
      # compared to a non-dtype string.
      self._dtype_policy = policy.Policy(dtype)
    elif dtype:
      self._dtype_policy = policy.Policy(dtypes.as_dtype(dtype).name)
    else:
      self._dtype_policy = policy.global_policy()
    if (self._dtype_policy.name == 'mixed_float16' and
        not loss_scale_optimizer.strategy_supports_loss_scaling()):
      # Although only loss scaling doesn't support certain strategies, to avoid
      # confusion, we disallow the 'mixed_float16' policy with unsupported
      # strategies. This is because 'mixed_float16' requires loss scaling for
      # numeric stability.
      strategy = distribute_lib.get_strategy()
      raise ValueError('Mixed precision is not supported with the '
                       'tf.distribute.Strategy: %s. Either stop using mixed '
                       'precision by removing the use of the "%s" policy or '
                       'use a different Strategy, e.g. a MirroredStrategy.' %
                       (strategy.__class__.__name__, self._dtype_policy.name))

    # Performance optimization: cache the compute dtype as a Dtype object or
    # None, so that str to Dtype conversion doesn't happen in Layer.__call__.
    # TODO(b/157486353): Investigate returning DTypes in Policy.
    if self._dtype_policy.compute_dtype:
      self._compute_dtype_object = dtypes.as_dtype(
          self._dtype_policy.compute_dtype)
    else:
      self._compute_dtype_object = None

  @property
  def dtype_policy(self):
    # [FUNCTION DOC]
    # Read-only property returning the layer's dtype policy object.
    """The dtype policy associated with this layer.

    This is an instance of a `tf.keras.mixed_precision.Policy`.
    """
    return self._dtype_policy

  @property
  def compute_dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the compute dtype name from the dtype policy (the dtype layer
    # computations and automatic input casts use).
    """The dtype of the layer's computations.

    This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
    mixed precision is used, this is the same as `Layer.dtype`, the dtype of
    the weights.

    Layers automatically cast their inputs to the compute dtype, which causes
    computations and the output to be in the compute dtype as well. This is done
    by the base Layer class in `Layer.__call__`, so you do not have to insert
    these casts if implementing your own layer.

    Layers often perform certain internal computations in higher precision when
    `compute_dtype` is float16 or bfloat16 for numeric stability. The output
    will still typically be float16 or bfloat16 in such cases.

    Returns:
      The layer's compute dtype.
    """
    return self._dtype_policy.compute_dtype

  @property
  def _compute_dtype(self):
    # [FUNCTION DOC]
    # HELPER: Backward-compatible private alias of the compute dtype property.
    # Returns the policy's compute dtype name.
    """Deprecated alias of `compute_dtype`."""
    return self._dtype_policy.compute_dtype

  @property
  def variable_dtype(self):
    # [FUNCTION DOC]
    # Read-only property, alias of the dtype property (the dtype of the layer's weights).
    """Alias of `Layer.dtype`, the dtype of the weights."""
    return self.dtype

  def _maybe_cast_inputs(self, inputs, input_list=None):
    # [FUNCTION DOC]
    # HELPER: Automatic input casting step of the layer call.
    # Inputs: inputs (tensor or nested structure) and input_list (optional flat list, computed if
    # missing). When autocast is enabled and the compute dtype is floating point, floating-point
    # tensors, sparse tensors and ragged tensors of a different floating dtype are cast to the
    # compute dtype. Returns the (possibly recast) inputs, unchanged object if nothing needed casting.
    """Maybe casts the inputs to the compute dtype.

    If self._compute_dtype is floating-point, and self_autocast is True,
    floating-point inputs are casted to self._compute_dtype.

    Args:
      inputs: Input tensor, or structure of input tensors.
      input_list: Flat list of input tensors.

    Returns:
      `inputs`, but tensors may have been casted to self._compute_dtype
    """
    if not input_list:
      input_list = nest.flatten(inputs)

    compute_dtype_object = self._compute_dtype_object
    should_autocast = (
        self._autocast and compute_dtype_object and
        compute_dtype_object.is_floating)

    if (should_autocast and
        any(map(self._should_cast_single_input, input_list))):
      # Only perform expensive `nest` operation when needed.
      return nest.map_structure(self._cast_single_input, inputs)
    else:
      return inputs

  def _should_cast_single_input(self, x):
    # [FUNCTION DOC]
    # HELPER: Predicate for automatic input casting.
    # Returns true when x is a tensor, sparse tensor or ragged tensor with a floating dtype that
    # differs from the layer's compute dtype; false otherwise (including other types).
    if isinstance(x, _AUTOCAST_TYPES):
      return (self._compute_dtype_object and
              x.dtype != self._compute_dtype_object and x.dtype.is_floating)
    return False

  def _cast_single_input(self, x):
    # [FUNCTION DOC]
    # HELPER: Mapping callback for automatic input casting.
    # Casts x to the compute dtype if the cast predicate says so, otherwise returns x unchanged.
    """Cast a single Tensor or TensorSpec to the compute dtype."""
    if self._should_cast_single_input(x):
      return math_ops.cast(x, self._compute_dtype_object)
    else:
      return x

  # _dtype used to be an attribute set in the constructor. We still expose it
  # because some clients still use it.
  # TODO(reedwm): Deprecate, then remove the _dtype property.
  @property
  def _dtype(self):
    # [FUNCTION DOC]
    # HELPER: Legacy private attribute kept for clients that still read it.
    # Property returning the variable dtype of the dtype policy.
    # This is equivalent to returning self.dtype . We do not return self.dtype
    # as it would cause infinite recursion in a few subclasses, which override
    # "dtype" to return self._dtype.
    return self._dtype_policy.variable_dtype

  @_dtype.setter
  def _dtype(self, value):
    # [FUNCTION DOC]
    # HELPER: Legacy private attribute setter.
    # Converts the given dtype-like value to a dtype name and replaces the layer's dtype policy
    # with a policy of that name. Returns nothing.
    value = dtypes.as_dtype(value).name
    self._set_dtype_policy(policy.Policy(value))

  def _name_scope(self):  # pylint: disable=method-hidden
    # [FUNCTION DOC]
    # HELPER: Computes the name scope string used while the layer is called in graph mode.
    # Returns the plain layer name when TF2 behavior is disabled; otherwise the layer name prefixed
    # with the current name scope and ending in a slash, which prevents numeric suffixes from
    # being appended. Takes no arguments, no side effects.
    if not tf2.enabled():
      return self.name
    name_scope = self.name
    current_name_scope = ops.get_name_scope()
    if current_name_scope:
      name_scope = current_name_scope + '/' + name_scope
    if name_scope:
      # Note that the trailing `/` prevents autogenerated
      # numerical suffixes to get appended. It will also fully reset
      # nested name scope (i.e. the outer name scope has no effect).
      name_scope += '/'
    return name_scope

  def _init_set_name(self, name, zero_based=True):
    # [FUNCTION DOC]
    # HELPER: Called by the constructor to assign the layer name.
    # Inputs: name (string or empty) and zero_based (bool, default true; whether auto-generated
    # numeric suffixes start at zero). With no name, generates a unique snake-case name from the
    # class name via the backend; otherwise registers the given name with the backend's name
    # observer and uses it. Stores the name on the layer; returns nothing.
    if not name:
      self._name = backend.unique_object_name(
          generic_utils.to_snake_case(self.__class__.__name__),
          zero_based=zero_based)
    else:
      backend.observe_object_name(name)
      self._name = name

  def _get_existing_metric(self, name=None):
    # [FUNCTION DOC]
    # HELPER: Lookup used by add_metric.
    # Input: name, string (default nothing). Returns the metric in this layer's own metric list with
    # that name, or nothing if none exists. Raises a value error if more than one metric has the name.
    match = [m for m in self._metrics if m.name == name]
    if not match:
      return
    if len(match) > 1:
      raise ValueError(
          'Please provide different names for the metrics you have added. '
          'We found {} metrics with the name: "{}"'.format(len(match), name))
    return match[0]

  def _handle_weight_regularization(self, name, variable, regularizer):
    # [FUNCTION DOC]
    # HELPER: Called by add_weight when a regularizer is given.
    # Inputs: name (variable name used for the name scope), variable (a variable or partitioned
    # variable) and regularizer (callable). Registers, via add_loss, one lazily evaluated
    # regularization loss per variable (per partition for split variables). Returns nothing.
    """Create lambdas which compute regularization losses."""

    def _loss_for_variable(v):
      # [FUNCTION DOC]
      # HELPER: Callback created for add_loss in weight regularization handling.
      # Applies the regularizer to variable v inside a name scope ending in Regularizer and
      # returns the resulting loss tensor.
      """Creates a regularization loss `Tensor` for variable `v`."""
      with backend.name_scope(name + '/Regularizer'):
        regularization = regularizer(v)
      return regularization

    if base_layer_utils.is_split_variable(variable):
      for v in variable:
        self.add_loss(functools.partial(_loss_for_variable, v))
    else:
      self.add_loss(functools.partial(_loss_for_variable, variable))

  def _handle_activity_regularization(self, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: Post-processing step of the layer call.
    # Inputs: inputs (unused) and outputs (tensor or nested structure). If an activity regularizer
    # is set, applies it to each flattened output, divides the loss by the output's batch size
    # (to make strength batch-agnostic) and registers it with add_loss. Returns nothing.
    # Apply activity regularization.
    # Note that it should be applied every time the layer creates a new
    # output, since it is output-specific.
    if self._activity_regularizer:
      output_list = nest.flatten(outputs)
      with backend.name_scope('ActivityRegularizer'):
        for output in output_list:
          activity_loss = self._activity_regularizer(output)
          batch_size = math_ops.cast(
              array_ops.shape(output)[0], activity_loss.dtype)
          # Make activity regularization strength batch-agnostic.
          mean_activity_loss = activity_loss / batch_size
          self.add_loss(mean_activity_loss)

  def _set_mask_metadata(self, inputs, outputs, previous_mask, build_graph):
    # [FUNCTION DOC]
    # HELPER: Post-processing step of the layer call that attaches output masks.
    # Inputs: inputs, outputs, previous_mask (mask(s) of the inputs) and build_graph (bool, whether
    # in graph building mode). Does nothing if the layer does not support masking. If all outputs
    # already carry masks (or the layer computes output and mask jointly) it only marks history as
    # checked when building a graph. Otherwise it calls compute_mask and stores each resulting mask
    # on the matching output tensor (silently skipping objects that cannot take attributes).
    # Returns nothing.
    # Many `Layer`s don't need to call `compute_mask`.
    # This method is optimized to do as little work as needed for the common
    # case.
    if not self._supports_masking:
      return

    flat_outputs = nest.flatten(outputs)

    mask_already_computed = (
        getattr(self, '_compute_output_and_mask_jointly', False) or
        all(getattr(x, '_keras_mask', None) is not None for x in flat_outputs))
    if mask_already_computed:
      if build_graph:
        self._set_mask_keras_history_checked(flat_outputs)
      return

    output_masks = self.compute_mask(inputs, previous_mask)
    if output_masks is None:
      return

    flat_masks = nest.flatten(output_masks)
    for tensor, mask in zip(flat_outputs, flat_masks):
      try:
        tensor._keras_mask = mask
      except AttributeError:
        # C Type such as np.ndarray.
        pass

    if build_graph:
      self._set_mask_keras_history_checked(flat_outputs)

  def _set_mask_keras_history_checked(self, flat_outputs):
    # [FUNCTION DOC]
    # HELPER: Used by mask metadata handling.
    # Input: flat_outputs, list of output tensors. For each output that has a mask, marks the mask
    # as history-checked so that mask tensors are not tracked when building op layers. Returns nothing.
    for output in flat_outputs:
      if getattr(output, '_keras_mask', None) is not None:
        # Do not track masks for `TensorFlowOpLayer` construction.
        output._keras_mask._keras_history_checked = True

  def _get_input_masks(self, inputs, input_list, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: Mask resolution step of the layer call.
    # Inputs: inputs (structure), input_list (flat list of inputs), args and kwargs (call
    # arguments). Returns a pair: the input masks and a flag saying whether they were found
    # implicitly. If the layer neither supports masking nor expects a mask argument, returns
    # nothing and false. If a mask argument was passed explicitly, returns it with false.
    # Otherwise gathers the masks attached to the input tensors; if all are empty returns nothing
    # and false, else returns them packed in the structure of the inputs and true.
    if not self._supports_masking and not self._expects_mask_arg:
      # Input masks only need to be retrieved if they are needed for `call`
      # or `compute_mask`.
      input_masks = None
      implicit_mask = False
    elif self._call_arg_was_passed('mask', args, kwargs):
      input_masks = self._get_call_arg_value('mask', args, kwargs)
      implicit_mask = False
    else:
      input_masks = [getattr(t, '_keras_mask', None) for t in input_list]
      if all(mask is None for mask in input_masks):
        input_masks = None
        implicit_mask = False
      else:
        # Only do expensive `nest` op when masking is actually being used.
        input_masks = nest.pack_sequence_as(inputs, input_masks)
        implicit_mask = True
    return input_masks, implicit_mask

  def _call_arg_was_passed(self, arg_name, args, kwargs, inputs_in_args=False):
    # [FUNCTION DOC]
    # HELPER: Introspection of how call arguments were supplied.
    # Inputs: arg_name (name of a parameter of call), args and kwargs (the call arguments), and
    # inputs_in_args (bool, default false; whether the first positional entry is the inputs
    # argument). Returns true if the argument appears in the keyword arguments or is supplied
    # positionally, false otherwise (and quickly false when nothing was passed).
    # Performance optimization: do no work in most common case.
    if not args and not kwargs:
      return False

    if arg_name in kwargs:
      return True
    call_fn_args = self._call_fn_args
    if not inputs_in_args:
      # Ignore `inputs` arg.
      call_fn_args = call_fn_args[1:]
    return arg_name in dict(zip(call_fn_args, args))

  def _get_call_arg_value(self, arg_name, args, kwargs, inputs_in_args=False):
    # [FUNCTION DOC]
    # HELPER: Introspection of how call arguments were supplied.
    # Inputs as for the was-passed check. Returns the value given for arg_name, from the keyword
    # arguments or by matching its position in the positional arguments. Raises a key error if the
    # argument was supplied in neither place.
    if arg_name in kwargs:
      return kwargs[arg_name]
    call_fn_args = self._call_fn_args
    if not inputs_in_args:
      # Ignore `inputs` arg.
      call_fn_args = call_fn_args[1:]
    args_dict = dict(zip(call_fn_args, args))
    return args_dict[arg_name]

  def _set_call_arg_value(
      self, arg_name, new_value, args,
      kwargs, inputs_in_args=False, pop_kwarg_if_none=False):
    # [FUNCTION DOC]
    # HELPER: Rewrites call arguments (used to inject or remove training and mask values).
    # Inputs: arg_name, new_value, args (positional), kwargs (keyword mapping), inputs_in_args (bool,
    # default false) and pop_kwarg_if_none (bool, default false). If the argument is currently given
    # positionally it is replaced in a copied tuple; otherwise it is stored in (or, when the new value
    # is nothing and pop_kwarg_if_none is true, removed from) the keyword mapping, which is mutated
    # in place. Returns the pair of resulting args and kwargs.
    arg_pos = self._call_fn_arg_positions.get(arg_name, None)
    if arg_pos is not None:
      if not inputs_in_args:
        # Ignore `inputs` arg.
        arg_pos = arg_pos - 1
      if len(args) > arg_pos:
        args = list(args)
        args[arg_pos] = new_value
        return tuple(args), kwargs
    if new_value is None and pop_kwarg_if_none:
      kwargs.pop(arg_name, None)
    else:
      kwargs[arg_name] = new_value
    return args, kwargs

  def _set_connectivity_metadata(self, args, kwargs, outputs):
    # [FUNCTION DOC]
    # HELPER: Functional-API construction glue.
    # Inputs: args and kwargs (all call arguments including inputs) and outputs (symbolic output
    # structure). Outputs that are identical objects to inputs are replaced by identity copies so
    # history metadata is not lost. Then creates a graph node connecting this layer to its inbound
    # and outbound layers (the node constructor updates the layer's node lists and sets history
    # metadata on the outputs). Returns the possibly copied outputs.
    # If the layer returns tensors from its inputs unmodified,
    # we copy them to avoid loss of KerasHistory metadata.
    flat_outputs = nest.flatten(outputs)
    flat_inputs = nest.flatten((args, kwargs))
    input_ids_set = {id(i) for i in flat_inputs}
    outputs_copy = []
    for x in flat_outputs:
      if id(x) in input_ids_set:
        with backend.name_scope(self.name):
          x = array_ops.identity(x)
      outputs_copy.append(x)
    outputs = nest.pack_sequence_as(outputs, outputs_copy)

    # Create node, Node wires itself to inbound and outbound layers.
    # The Node constructor actually updates this layer's self._inbound_nodes,
    # sets _keras_history on the outputs, and adds itself to the
    # `_outbound_nodes` of the layers that produced the inputs to this
    # layer call.
    node_module.Node(self, call_args=args, call_kwargs=kwargs, outputs=outputs)
    return outputs

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    # [FUNCTION DOC]
    # HELPER: Shared implementation of the get-input/output/shape-at-node methods.
    # KNOWN ISSUE: The docstring says it raises a runtime error in eager mode; the code has no eager
    # check (the runtime error is only for a never-called layer, and negative indices are accepted).
    # Inputs: node_index (integer), attr (node attribute name), attr_name (readable name for error
    # messages). Raises a runtime error if there are no inbound nodes and a value error if the index
    # is too large. Returns the node's attribute, unwrapping a one-element list to its element.
    """Private utility to retrieves an attribute (e.g. inputs) from a node.

    This is used to implement the methods:
        - get_input_shape_at
        - get_output_shape_at
        - get_input_at
        etc...

    Args:
        node_index: Integer index of the node from which
            to retrieve the attribute.
        attr: Exact node attribute name.
        attr_name: Human-readable attribute name, for error messages.

    Returns:
        The layer's attribute `attr` at the node of index `node_index`.

    Raises:
        RuntimeError: If the layer has no inbound nodes, or if called in Eager
        mode.
        ValueError: If the index provided does not match any node.
    """
    if not self._inbound_nodes:
      raise RuntimeError('The layer has never been called '
                         'and thus has no defined ' + attr_name + '.')
    if not len(self._inbound_nodes) > node_index:
      raise ValueError('Asked to get ' + attr_name + ' at node ' +
                       str(node_index) + ', but the layer has only ' +
                       str(len(self._inbound_nodes)) + ' inbound nodes.')
    values = getattr(self._inbound_nodes[node_index], attr)
    if isinstance(values, list) and len(values) == 1:
      return values[0]
    else:
      return values

  def _maybe_build(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Lazy-build step of the layer call.
    # Input: inputs, a structure of tensors or shapes. If the layer is not built: checks input
    # spec compatibility, infers the dtype policy from the first input when the compute dtype is
    # unset, derives tensor shapes from the inputs, and (only when build is overridden) calls build
    # inside an init scope; then always marks the layer built through the base build. Afterwards, if
    # initial weights were given at construction, assigns them via set_weights in an init scope
    # and clears them. Returns nothing.
    # Check input assumptions set before layer building, e.g. input rank.
    if not self.built:
      input_spec.assert_input_compatibility(
          self.input_spec, inputs, self.name)
      input_list = nest.flatten(inputs)
      if input_list and self._dtype_policy.compute_dtype is None:
        try:
          dtype = input_list[0].dtype.base_dtype.name
        except AttributeError:
          pass
        else:
          self._set_dtype_policy(policy.Policy(dtype))
      input_shapes = None
      # Converts Tensors / CompositeTensors to TensorShapes.
      if all(hasattr(x, 'shape') for x in input_list):
        input_shapes = tf_utils.get_shapes(inputs)
      else:
        # Converts input shape to TensorShapes.
        try:
          input_shapes = tf_utils.convert_shapes(inputs, to_tuples=False)
        except ValueError:
          pass
      # Only call `build` if the user has manually overridden the build method.
      if not hasattr(self.build, '_is_default'):
        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf_utils.maybe_init_scope(self):
          self.build(input_shapes)  # pylint:disable=not-callable
      # We must set also ensure that the layer is marked as built, and the build
      # shape is stored since user defined build functions may not be calling
      # `super.build()`
      Layer.build(self, input_shapes)

    # Optionally load weight values specified at layer instantiation.
    if self._initial_weights is not None:
      with ops.init_scope():
        # Using `init_scope` since we want variable assignment in
        # `set_weights` to be treated like variable initialization.
        self.set_weights(self._initial_weights)
      self._initial_weights = None

  def _symbolic_call(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Shape-based symbolic output construction (legacy path).
    # Input: inputs, structure of tensors. Takes their shapes, asks compute_output_shape for the
    # output shapes, converts them to tensor shape objects and returns a structure of placeholders
    # of those shapes with the layer's dtype. No call is executed.
    input_shapes = nest.map_structure(lambda x: x.shape, inputs)
    output_shapes = self.compute_output_shape(input_shapes)
    # Convert to TensorShape so that nest.map_structure will not map into
    # individual dim of the shape.
    output_shapes = tf_utils.convert_shapes(output_shapes, to_tuples=False)

    def _make_placeholder_like(shape):
      # [FUNCTION DOC]
      # HELPER: Mapping callback of the symbolic call helper.
      # Creates a backend placeholder of the given shape and the layer's dtype with its mask
      # attribute cleared, and returns it.
      ph = backend.placeholder(shape=shape, dtype=self.dtype)
      ph._keras_mask = None
      return ph
    return nest.map_structure(_make_placeholder_like, output_shapes)

  def _get_trainable_state(self):
    # [FUNCTION DOC]
    # HELPER: Used with the set-trainable-state counterpart to temporarily change trainability.
    # Returns a weak-key mapping from this layer and every nested sub-layer to its current
    # trainable flag.
    """Get the `trainable` state of each sublayer.

    Returns:
      A dict mapping all sublayers to their `trainable` value.
    """
    trainable_state = weakref.WeakKeyDictionary()
    for layer in self._flatten_layers():
      trainable_state[layer] = layer.trainable
    return trainable_state

  def _set_trainable_state(self, trainable_state):
    # [FUNCTION DOC]
    # HELPER: Restores trainability recorded by the get-trainable-state counterpart.
    # Input: trainable_state, mapping from layers to booleans. For this layer and each nested
    # sub-layer present in the mapping, sets its trainable flag. Returns nothing.
    """Set `trainable` state for each sublayer."""
    for layer in self._flatten_layers():
      if layer in trainable_state:
        layer.trainable = trainable_state[layer]

  @property
  def _obj_reference_counts(self):
    # [FUNCTION DOC]
    # HELPER: Attribute-tracking bookkeeping for attribute assignment and deletion.
    # Property returning an identity-keyed mapping from objects to the number of this layer's
    # attributes that reference them; creates the mapping on first access.
    """A dictionary counting the number of attributes referencing an object."""
    self._maybe_create_attribute('_obj_reference_counts_dict',
                                 object_identity.ObjectIdentityDictionary())
    return self._obj_reference_counts_dict

  @trackable.no_automatic_dependency_tracking
  def _maybe_create_attribute(self, name, default_value):
    # [FUNCTION DOC]
    # HELPER: Used by the constructor and attribute tracking to create tracking fields safely.
    # Inputs: name (string) and default_value (any object). Sets the attribute to the default only
    # if the layer does not already have it (without automatic dependency tracking). Returns nothing.
    """Create the attribute with the default value if it hasn't been created.

    This is useful for fields that is used for tracking purpose,
    _trainable_weights, or _layers. Note that user could create a layer subclass
    and assign an internal field before invoking the Layer.__init__(), the
    __setattr__() need to create the tracking fields and __init__() need to not
    override them.

    Args:
      name: String, the name of the attribute.
      default_value: Object, the default value of the attribute.
    """
    if not hasattr(self, name):
      self.__setattr__(name, default_value)

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Attribute deletion with tracking cleanup. Input: name of the attribute to delete. Looks up the
    # current value and its reference count among the layer's attributes. If other attributes still
    # reference the object, only decrements the count and deletes the attribute. If it was the last
    # reference, also removes the object from the tracked sub-layer/trackable list (for layers or
    # objects with weights) and, for variables, from the trainable and non-trainable weight lists.
    # The base tracking deletion is invoked while skipping the automatic-tracking class. Returns nothing.
    # For any super.__delattr__() call, we will directly use the implementation
    # in Trackable and skip the behavior in AutoTrackable. The Layer was
    # originally use Trackable as base class, the change of using Module as base
    # class forced us to have AutoTrackable in the class hierarchy.
    #
    # TODO(b/180760306) Keeping the status quo of skipping _delattr__ and
    # __setattr__ in AutoTrackable may be unsustainable.
    existing_value = getattr(self, name, None)

    # If this value is replacing an existing object assigned to an attribute, we
    # should clean it out to avoid leaking memory. First we check if there are
    # other attributes referencing it.
    reference_counts = self._obj_reference_counts
    if existing_value not in reference_counts:
      super(autotrackable.AutoTrackable, self).__delattr__(name)  # pylint: disable=bad-super-call
      return

    reference_count = reference_counts[existing_value]
    if reference_count > 1:
      # There are other remaining references. We can't remove this object from
      # _layers etc.
      reference_counts[existing_value] = reference_count - 1
      super(autotrackable.AutoTrackable, self).__delattr__(name)  # pylint: disable=bad-super-call
      return
    else:
      # This is the last remaining reference.
      del reference_counts[existing_value]

    super(autotrackable.AutoTrackable, self).__delattr__(name)  # pylint: disable=bad-super-call

    if (isinstance(existing_value, Layer)
        or base_layer_utils.has_weights(existing_value)):
      super(autotrackable.AutoTrackable, self).__setattr__(  # pylint: disable=bad-super-call
          '_self_tracked_trackables',
          [l for l in self._self_tracked_trackables if l is not existing_value])
    if isinstance(existing_value, tf_variables.Variable):
      super(autotrackable.AutoTrackable, self).__setattr__(  # pylint: disable=bad-super-call
          '_trainable_weights',
          [w for w in self._trainable_weights if w is not existing_value])
      super(autotrackable.AutoTrackable, self).__setattr__(  # pylint: disable=bad-super-call
          '_non_trainable_weights',
          [w for w in self._non_trainable_weights if w is not existing_value])

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Attribute assignment with automatic tracking. Inputs: name and value. If tracking is
    # disabled, or the class defines the name (such as a property setter), the value is set
    # directly (an attribute error with a clearer message is raised if it conflicts with a
    # read-only property). Otherwise: wraps data structures for tracking, updates the reference
    # count, deletes any previous value (cleaning tracking lists), records metric instances in the
    # metrics list, appends sub-layers/modules/objects with weights to the tracked list (marking
    # legacy layers to use resource variables when applicable), adds any variables found
    # (including inside nested structures) to the trainable or non-trainable list and registers them
    # with the backend, and finally stores the attribute. Returns nothing.
    if (name == '_self_setattr_tracking' or
        not getattr(self, '_self_setattr_tracking', True) or
        # Exclude @property.setters from tracking
        hasattr(self.__class__, name)):
      try:
        super(autotrackable.AutoTrackable, self).__setattr__(name, value)  # pylint: disable=bad-super-call
      except AttributeError:
        raise AttributeError(
            ('Can\'t set the attribute "{}", likely because it conflicts with '
             'an existing read-only @property of the object. Please choose a '
             'different name.').format(name))
      return

    # Wraps data structures in `Trackable`, unwraps `NoDependency` objects.
    value = data_structures.sticky_attribute_assignment(
        trackable=self, value=value, name=name)

    reference_counts = self._obj_reference_counts
    reference_counts[value] = reference_counts.get(value, 0) + 1

    # Clean out the old attribute, which clears _layers and _trainable_weights
    # if necessary.
    try:
      self.__delattr__(name)
    except AttributeError:
      pass

    # Keep track of metric instance created in subclassed layer.
    for val in nest.flatten(value):
      if isinstance(val, metrics_mod.Metric) and hasattr(self, '_metrics'):
        self._metrics.append(val)

    # Append value to self._self_tracked_trackables if relevant
    if (getattr(self, '_auto_track_sub_layers', True) and
        (isinstance(value, module.Module) or
         base_layer_utils.has_weights(value))):
      self._maybe_create_attribute('_self_tracked_trackables', [])
      # We need to check object identity to avoid de-duplicating empty
      # container types which compare equal.
      if not any((layer is value for layer in self._self_tracked_trackables)):
        self._self_tracked_trackables.append(value)
        if hasattr(value, '_use_resource_variables'):
          # Legacy layers (V1 tf.layers) must always use
          # resource variables.
          value._use_resource_variables = True

    # Append value to list of trainable / non-trainable weights if relevant
    # TODO(b/125122625): This won't pick up on any variables added to a
    # list/dict after creation.
    for val in nest.flatten(value, expand_composites=True):
      if not isinstance(val, tf_variables.Variable):
        continue

      # Users may add extra weights/variables
      # simply by assigning them to attributes (invalid for graph networks)
      self._maybe_create_attribute('_trainable_weights', [])
      self._maybe_create_attribute('_non_trainable_weights', [])
      if val.trainable:
        if any(val is w for w in self._trainable_weights):
          continue
        self._trainable_weights.append(val)
      else:
        if any(val is w for w in self._non_trainable_weights):
          continue
        self._non_trainable_weights.append(val)

      backend.track_variable(val)

    # TODO(b/180760306) Skip the auto trackable from tf.Module to keep status
    # quo. See the comment at __delattr__.
    super(autotrackable.AutoTrackable, self).__setattr__(name, value)  # pylint: disable=bad-super-call

  def _gather_children_attribute(self, attribute):
    # [FUNCTION DOC]
    # HELPER: Used by the weight properties to include sub-layer weights.
    # Input: attribute, one of the names for variables, trainable variables or non-trainable
    # variables (anything else fails an assertion). Returns a concatenated list of that attribute
    # from each direct child module (non-recursive), or an empty list if the layer tracks nothing.
    assert attribute in {
        'variables', 'trainable_variables', 'non_trainable_variables'
    }
    if hasattr(self, '_self_tracked_trackables'):
      nested_layers = self._flatten_modules(include_self=False, recursive=False)
      return list(
          itertools.chain.from_iterable(
              getattr(layer, attribute) for layer in nested_layers))
    return []

  def _flatten_layers(self, recursive=True, include_self=True):
    # [FUNCTION DOC]
    # HELPER: Traversal utility used by many layer properties.
    # Inputs: recursive (bool, default true; descend into nested sub-layers) and include_self
    # (bool, default true). Generator yielding this layer (optionally) and the tracked modules that
    # are layers, in tracking order.
    for m in self._flatten_modules(
        recursive=recursive, include_self=include_self):
      if isinstance(m, Layer):
        yield m

  def _flatten_modules(self, recursive=True, include_self=True):
    # [FUNCTION DOC]
    # HELPER: Traversal utility underlying the layer flattening helper.
    # Inputs: recursive (bool, default true) and include_self (bool, default true). Generator
    # yielding this layer (optionally), then every tracked module that is not a metric, each once
    # (deduplicated by identity), in breadth-ordered tracking order; for data structures it looks
    # inside their values even when not recursive.
    """Flattens `tf.Module` instances (excluding `Metrics`).

    Args:
      recursive: Whether to recursively flatten through submodules.
      include_self: Whether to include this `Layer` instance.

    Yields:
      `tf.Module` instance tracked by this `Layer`.
    """
    if include_self:
      yield self

    # Only instantiate set and deque if needed.
    trackables = getattr(self, '_self_tracked_trackables', None)
    if trackables:
      seen_object_ids = set()
      deque = collections.deque(trackables)
      while deque:
        trackable_obj = deque.popleft()
        trackable_id = id(trackable_obj)
        if trackable_id in seen_object_ids:
          continue
        seen_object_ids.add(trackable_id)

        # Metrics are not considered part of the Layer's topology.
        if (isinstance(trackable_obj, module.Module) and
            not isinstance(trackable_obj, metrics_mod.Metric)):
          yield trackable_obj
          # Introspect recursively through sublayers.
          if recursive:
            subtrackables = getattr(trackable_obj, '_self_tracked_trackables',
                                    None)
            if subtrackables:
              deque.extendleft(reversed(subtrackables))
        elif isinstance(trackable_obj, data_structures.TrackableDataStructure):
          # Data structures are introspected even with `recursive=False`.
          tracked_values = trackable_obj._values
          if tracked_values:
            deque.extendleft(reversed(tracked_values))

  # This is a hack so that the is_layer (within
  # training/trackable/layer_utils.py) check doesn't get the weights attr.
  # TODO(b/110718070): Remove when fixed.
  def _is_layer(self):
    # [FUNCTION DOC]
    # HELPER: Marker method used by layer-detection utilities (so they avoid reading weights).
    # Always returns true.
    return True

  def _init_call_fn_args(self, expects_training_arg=None):
    # [FUNCTION DOC]
    # HELPER: Inspects the call method signature; used by the constructor and by saved-model loading.
    # KNOWN ISSUE: The list of call argument names is obtained from a cached property and keyword-only
    # names are appended with an in-place operation, so the cached list itself is mutated (the
    # keyword-only names become part of the cached argument list and positions are unaffected only
    # because they are appended at the end).
    # Input: expects_training_arg, optional bool; when nothing it is inferred from whether call
    # takes a training argument or arbitrary keywords; otherwise the given value is used. Clears
    # the cached signature information, then sets whether the layer expects training and mask
    # arguments and the default value of the training argument (nothing if no non-empty default).
    # Clear cached call function arguments.
    self.__class__._call_full_argspec.fget.cache.pop(self, None)
    self.__class__._call_fn_args.fget.cache.pop(self, None)
    self.__class__._call_accepts_kwargs.fget.cache.pop(self, None)

    call_fn_args = self._call_fn_args
    call_fn_args += self._call_full_argspec.kwonlyargs or []
    if expects_training_arg is None:
      self._expects_training_arg = ('training' in call_fn_args or
                                    self._call_accepts_kwargs)
    else:
      # Use value encoded into the metadata when loading from the SavedModel.
      self._expects_training_arg = expects_training_arg
    # The default training arg will be any (non-None) default specified in the
    # method signature, or None if no value is specified.
    call_fn_arg_defaults = self._call_fn_arg_defaults.copy()
    call_fn_arg_defaults.update(self._call_full_argspec.kwonlydefaults or {})
    self._default_training_arg = call_fn_arg_defaults.get('training')

    self._expects_mask_arg = ('mask' in call_fn_args or
                              self._call_accepts_kwargs)

  @property
  @layer_utils.cached_per_instance
  def _call_full_argspec(self):
    # [FUNCTION DOC]
    # HELPER: Cached (per instance) property returning the full argument specification of call.
    # Argspec inspection is expensive and the call spec is used often, so it
    # makes sense to cache the result.
    return tf_inspect.getfullargspec(self.call)

  @property
  @layer_utils.cached_per_instance
  def _call_fn_args(self):
    # [FUNCTION DOC]
    # HELPER: Cached (per instance) property returning the positional parameter names of call,
    # without a leading self (which can appear when call is wrapped by a decorator).
    all_args = self._call_full_argspec.args
    # Scrub `self` that appears if a decorator was applied.
    if all_args and all_args[0] == 'self':
      return all_args[1:]
    return all_args

  @property
  @layer_utils.cached_per_instance
  def _call_fn_arg_defaults(self):
    # [FUNCTION DOC]
    # HELPER: Cached (per instance) property returning a mapping from the names of call parameters
    # that have defaults to those default values (defaults belong to the last parameters).
    call_fn_args = self._call_fn_args
    call_fn_defaults = self._call_full_argspec.defaults or []
    defaults = dict()

    # The call arg defaults are an n-tuple of the last n elements of the args
    # list. (n = # of elements that have a default argument)
    for i in range(-1 * len(call_fn_defaults), 0):
      defaults[call_fn_args[i]] = call_fn_defaults[i]
    return defaults

  @property
  @layer_utils.cached_per_instance
  def _call_fn_arg_positions(self):
    # [FUNCTION DOC]
    # HELPER: Cached (per instance) property returning a mapping from call parameter names to their
    # zero-based positions (the inputs parameter is position zero).
    call_fn_arg_positions = dict()
    for pos, arg in enumerate(self._call_fn_args):
      call_fn_arg_positions[arg] = pos
    return call_fn_arg_positions

  @property
  @layer_utils.cached_per_instance
  def _call_accepts_kwargs(self):
    # [FUNCTION DOC]
    # HELPER: Cached (per instance) property: true if call accepts arbitrary keyword arguments.
    return self._call_full_argspec.varkw is not None

  @property
  def _eager_losses(self):
    # [FUNCTION DOC]
    # HELPER: Loss bookkeeping for add_loss and the losses property.
    # Property returning the thread-local list of eager losses added during the current call,
    # creating an empty list for the current thread on first access.
    # A list of loss values containing activity regularizers and losses
    # manually added through `add_loss` during eager execution. It is cleared
    # after every batch.
    # Because we plan on eventually allowing a same model instance to be trained
    # in eager mode or graph mode alternatively, we need to keep track of
    # eager losses and symbolic losses via separate attributes.
    if not hasattr(self._thread_local, '_eager_losses'):
      self._thread_local._eager_losses = []
    return self._thread_local._eager_losses

  @_eager_losses.setter
  def _eager_losses(self, losses):
    # [FUNCTION DOC]
    # HELPER: Loss bookkeeping setter. Replaces the current thread's eager loss list with the given
    # list.
    self._thread_local._eager_losses = losses

  def _dedup_weights(self, weights):
    # [FUNCTION DOC]
    # HELPER: Used by the weight properties.
    # Input: weights, list of variables. Returns a new list with duplicates (by object identity)
    # removed, keeping first-occurrence order.
    """Dedupe weights while maintaining order as much as possible."""
    output, seen_ids = [], set()
    for w in weights:
      if id(w) not in seen_ids:
        output.append(w)
        # Track the Variable's identity to avoid __eq__ issues.
        seen_ids.add(id(w))

    return output

  def _split_out_first_arg(self, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: First step of the layer call wrapper.
    # Inputs: args (positional call arguments) and kwargs (keyword mapping). Takes the inputs from
    # the first positional argument, or else from the keyword named like call's first parameter
    # (on a copied mapping so the caller's mapping is untouched). Raises a value error if neither is
    # present. Returns a triple: inputs, remaining positional args, remaining keyword arguments.
    # Grab the argument corresponding to the first argument in the
    # layer's `call` method spec. This will either be the first positional
    # argument, or it will be provided as a keyword argument.
    if args:
      inputs = args[0]
      args = args[1:]
    elif self._call_fn_args[0] in kwargs:
      kwargs = copy.copy(kwargs)
      inputs = kwargs.pop(self._call_fn_args[0])
    else:
      raise ValueError(
          'The first argument to `Layer.call` must always be passed.')
    return inputs, args, kwargs

  # SavedModel properties. Please see keras/saving/saved_model for details.

  @trackable.no_automatic_dependency_tracking
  def _set_save_spec(self, inputs):
    # [FUNCTION DOC]
    # HELPER: SavedModel support.
    # Input: inputs, structure of tensors from the first call. Records tensor specs of the inputs as
    # the saved-model input spec unless one is already recorded. Returns nothing.
    if self._saved_model_inputs_spec is not None:
      return  # Already set.

    self._saved_model_inputs_spec = nest.map_structure(tf_utils.get_tensor_spec,
                                                       inputs)

  def _get_save_spec(self, dynamic_batch=True):
    # [FUNCTION DOC]
    # HELPER: SavedModel support.
    # Input: dynamic_batch, bool default true (whether the batch dimension is made unknown).
    # Returns the recorded input specs rebuilt with that batch setting, or nothing if no spec was
    # recorded yet.
    if self._saved_model_inputs_spec is None:
      return None

    return nest.map_structure(
        lambda t: tf_utils.get_tensor_spec(t, dynamic_batch=dynamic_batch),
        self._saved_model_inputs_spec)

  @property
  def _trackable_saved_model_saver(self):
    # [FUNCTION DOC]
    # HELPER: SavedModel support. Property returning a new layer saved-model saver object for this
    # layer (a new one on each access).
    return layer_serialization.LayerSavedModelSaver(self)

  @property
  def _object_identifier(self):
    # [FUNCTION DOC]
    # HELPER: SavedModel support. Property returning the object identifier string the saver
    # assigns to this layer.
    return self._trackable_saved_model_saver.object_identifier

  @property
  def _tracking_metadata(self):
    # [FUNCTION DOC]
    # HELPER: SavedModel support. Property returning the metadata (serialized text) the saver
    # stores for this layer.
    """Info about this layer to be saved into the SavedModel."""
    return self._trackable_saved_model_saver.tracking_metadata

  def _trackable_children(self, save_type='checkpoint', **kwargs):
    # [FUNCTION DOC]
    # HELPER: Checkpoint/SavedModel dependency enumeration override.
    # Inputs: save_type (string, default checkpoint; the value savedmodel adds layer-specific
    # children) and keyword arguments, which must include a cache mapping when saving a
    # SavedModel (a missing one raises a key error). Returns a mapping from child names to
    # trackable objects: the saver's children first (SavedModel only), updated with the parent
    # class's children.
    if save_type == 'savedmodel':
      cache = kwargs['cache']
      # TODO(b/213628533): This must be called before super() to ensure
      # that any input shape changes are applied before getting the config of
      # the model.
      children = self._trackable_saved_model_saver.trackable_children(cache)
    else:
      children = {}
    children.update(super()._trackable_children(save_type, **kwargs))
    return children

  @property
  def _use_input_spec_as_call_signature(self):
    # [FUNCTION DOC]
    # HELPER: SavedModel support. Property: true if this layer's class is exported under the keras
    # public API (so its input spec can be used as the call signature when tracing), else false.
    # Whether input spec can be used as the call signature when tracing the
    # Layer for SavedModel. By default, this is set to `True` for layers
    # exported from the Keras library, because the layers more rigidly define
    # the `input_specs` property (many custom layers only set the `ndims`)
    return get_canonical_name_for_symbol(type(self),
                                         api_name='keras') is not None

  def __getstate__(self):
    # [FUNCTION DOC]
    # Supports copying and pickling. Returns a shallow copy of the instance state without the
    # thread-local storage and the metrics lock (which cannot be copied).
    # Override to support `copy.deepcopy` and pickling.
    # Thread-local objects cannot be copied in Python 3, so pop these.
    # Thread-local objects are used to cache losses in MirroredStrategy, and
    # so shouldn't be copied.
    state = self.__dict__.copy()
    state.pop('_thread_local', None)
    state.pop('_metrics_lock', None)
    return state

  def __setstate__(self, state):
    # [FUNCTION DOC]
    # Restores instance state when copying or unpickling. Input: state, mapping of attributes. Adds
    # fresh thread-local storage and a fresh metrics lock to it and installs it as the instance
    # attribute dictionary, bypassing tracking logic. Returns nothing.
    state['_thread_local'] = threading.local()
    state['_metrics_lock'] = threading.Lock()
    # Bypass Trackable logic as `__dict__` already contains this info.
    object.__setattr__(self, '__dict__', state)


class TensorFlowOpLayer(Layer):
  # [CLASS DOC]
  # Layer that wraps a raw TensorFlow operation applied to symbolic tensors so it can take part in
  # a functional model. Holds the operation's node definition (parsed protocol message), a mapping
  # from input index to constant values for inputs that are not keras inputs, and is marked as built
  # and as needing restoration from its config in SavedModel loading. Each call recreates the
  # operation on the new inputs (inside a tf function when executing eagerly) and records its gradient.
  # Created internally by the functional API when ops lack keras metadata; name gets a fixed prefix.
  """Wraps a TensorFlow Operation in a Layer.

  This class is used internally by the Functional API. When a user
  uses a raw TensorFlow Operation on symbolic tensors originating
  from an `Input` Layer, the resultant operation will be wrapped
  with this Layer object in order to make the operation compatible
  with the Keras API.

  This Layer will create a new, identical operation (except for inputs
  and outputs) every time it is called. If `run_eagerly` is `True`,
  the op creation and calculation will happen inside an Eager function.

  Instances of this Layer are created when `autolambda` is called, which
  is whenever a Layer's `__call__` encounters symbolic inputs that do
  not have Keras metadata, or when a Network's `__init__` encounters
  outputs that do not have Keras metadata.

  Attributes:
    node_def: String, the serialized NodeDef of the Op this layer will wrap.
    name: String, the name of the Layer.
    constants: Dict of NumPy arrays, the values of any Tensors needed for this
      Operation that do not originate from a Keras `Input` Layer. Since all
      placeholders must come from Keras `Input` Layers, these Tensors must be
      treated as constant in the Functional API.
    trainable: Bool, whether this Layer is trainable. Currently Variables are
      not supported, and so this parameter has no effect.
    dtype: The default dtype of this Layer. Inherited from `Layer` and has no
      effect on this class, however is used in `get_config`.
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self,
               node_def,
               name,
               constants=None,
               trainable=True,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: node_def (the operation definition, either a mapping that is parsed into
    # a node definition message, or serialized bytes/text which is decoded), name (string, gets the op
    # layer prefix), constants (optional mapping from input index, possibly given as text keys after
    # JSON, to constant values; default empty), trainable (bool, default true; no real effect) and
    # dtype (optional). Calls the base constructor with autocast disabled so input types still match
    # the op, stores the node definition and constants, marks the layer built, and flags that it must
    # be restored from config in SavedModel loading.
    # Pass autocast=False, as if inputs are cast, input types might not match
    # Operation type.
    super(TensorFlowOpLayer, self).__init__(
        name=_TF_OP_LAYER_NAME_PREFIX + name, trainable=trainable, dtype=dtype,
        autocast=False)
    if isinstance(node_def, dict):
      self.node_def = json_format.ParseDict(node_def, node_def_pb2.NodeDef())
    else:
      if not isinstance(node_def, bytes):
        node_def = node_def.encode('utf-8')
      self.node_def = node_def_pb2.NodeDef.FromString(node_def)
    # JSON serialization stringifies keys which are integer input indices.
    self.constants = ({
        int(index): constant for index, constant in constants.items()
    } if constants is not None else {})
    # Layer uses original op unless it is called on new inputs.
    # This means `built` is not set in `__call__`.
    self.built = True

    # Do not individually trace TensorflowOpLayers in the SavedModel.
    self._must_restore_from_config = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Applies the wrapped operation. Input: inputs, tensor or structure of tensors. When executing
    # eagerly it runs the op creation inside a traced function; otherwise it creates the op directly
    # in the inputs' graph. Returns the op's single output tensor or list of outputs.
    if context.executing_eagerly():
      return self._defun_call(inputs)
    return self._make_op(inputs)

  def _make_node_def(self, graph):
    # [FUNCTION DOC]
    # HELPER: Part of op recreation. Input: graph, the target graph. Returns a copy of the stored
    # node definition marked as cloned (so TPU context does not add attributes) and renamed to a name
    # unique in that graph.
    node_def = node_def_pb2.NodeDef()
    node_def.CopyFrom(self.node_def)
    # Used in TPUReplicateContext to indicate whether this node has been cloned
    # and to not add TPU attributes.
    node_def.attr['_cloned'].b = True
    node_def.name = graph.unique_name(node_def.name)
    return node_def

  def _make_op(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Core of the op layer call. Input: inputs, tensor or structure whose first tensor
    # supplies the graph. Creates a fresh copy of the wrapped operation in that graph: re-creates
    # constant inputs as graph constants and inserts them at their recorded positions, builds the
    # op from the node definition using private graph APIs, runs control-flow post-processing,
    # records the op's gradient for backprop, and returns the single output tensor or the list of
    # outputs. Side effects: adds an op to the graph and records a gradient.
    inputs = nest.flatten(inputs)
    graph = inputs[0].graph
    node_def = self._make_node_def(graph)
    with graph.as_default():
      for index, constant in self.constants.items():
        # Recreate constant in graph to add distribution context.
        value = tensor_util.constant_value(constant)
        if value is not None:
          constant = constant_op.constant(value, name=node_def.input[index])
        inputs.insert(index, constant)
      # TODO(b/183990973): We should drop or consolidate these private api calls
      # for adding an op to the graph and recording its gradient.
      c_op = ops._create_c_op(graph, node_def, inputs, control_inputs=[])
      op = graph._create_op_from_tf_operation(c_op)
      op._control_flow_post_processing()

      # Record the gradient because custom-made ops don't go through the
      # code-gen'd eager call path
      op_type = compat.as_str(op.op_def.name)
      attr_names = [compat.as_str(attr.name) for attr in op.op_def.attr]
      attrs = []
      for attr_name in attr_names:
        attrs.append(attr_name)
        attrs.append(op.get_attr(attr_name))
      attrs = tuple(attrs)
      backprop.record_gradient(op_type, op.inputs, attrs, op.outputs)

      if len(op.outputs) == 1:
        return op.outputs[0]
      return op.outputs

  @def_function.function
  def _defun_call(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Eager-mode path of the op layer call (decorated as a traced function).
    # Input: inputs, tensors. Creates the op inside the traced function and returns its output(s).
    """Wraps the op creation method in an Eager function for `run_eagerly`."""
    return self._make_op(inputs)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer config with the name restored to the constructor argument (the op layer
    # prefix removed), the node definition converted to a plain mapping, and the constants converted
    # to concrete values read from the backend.
    config = super(TensorFlowOpLayer, self).get_config()
    config.update({
        # `__init__` prefixes the name. Revert to the constructor argument.
        'name': config['name'][len(_TF_OP_LAYER_NAME_PREFIX):],
        'node_def': json_format.MessageToDict(self.node_def),
        'constants': {
            i: backend.get_value(c) for i, c in self.constants.items()
        }
    })
    return config


class AddLoss(Layer):
  # [CLASS DOC]
  # Pass-through layer that registers its inputs as a loss of the model when called (used when a
  # loss is added to a functional model). Holds an unconditional flag; autocast is forced off so
  # losses keep their dtype. Returns its inputs unchanged.
  """Adds its inputs as a loss.

  Attributes:
    unconditional: Whether or not the loss should be conditioned on the inputs.
  """

  def __init__(self, unconditional, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: unconditional (bool, whether the loss is independent of the inputs) and
    # keyword arguments for the base layer (autocast is overridden to false). Stores the flag.
    # Pass autocast=False, as there is no reason to cast loss to a different
    # dtype.
    kwargs['autocast'] = False
    super(AddLoss, self).__init__(**kwargs)
    self.unconditional = unconditional

  def call(self, inputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Passes an inputs keyword to add_loss whose value is the negation of the
    # unconditional flag, but add_loss discards that keyword, so the flag has no effect.
    # Registers the inputs as loss(es) on this layer and returns the inputs unchanged.
    self.add_loss(inputs, inputs=(not self.unconditional))
    return inputs

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer config extended with the unconditional flag.
    config = super(AddLoss, self).get_config()
    config.update({'unconditional': self.unconditional})
    return config


class AddMetric(Layer):
  # [CLASS DOC]
  # Pass-through layer that registers its inputs as a named metric when called (used when a metric
  # is added to a functional model). Holds an aggregation setting and a metric name. Returns its
  # inputs unchanged.
  """Adds its inputs as a metric.

  Attributes:
    aggregation: 'mean' or None. How the inputs should be aggregated.
    metric_name: The name to use for this metric.
  """

  def __init__(self, aggregation=None, metric_name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: aggregation (optional, mean or nothing), metric_name (optional string)
    # and base layer keyword arguments. Stores both settings.
    super(AddMetric, self).__init__(**kwargs)
    self.aggregation = aggregation
    self.metric_name = metric_name

  def call(self, inputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Forwards the aggregation setting to add_metric, which validates but ignores it.
    # Registers the inputs as a metric with the stored name and returns the inputs unchanged.
    self.add_metric(inputs, aggregation=self.aggregation, name=self.metric_name)
    return inputs

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base layer config extended with the aggregation and metric name.
    config = super(AddMetric, self).get_config()
    config.update({
        'aggregation': self.aggregation,
        'metric_name': self.metric_name
    })
    return config


def _in_functional_construction_mode(layer, inputs, args, kwargs, input_list):  # pylint: disable=unused-argument
  # [FUNCTION DOC]
  # HELPER: Mode check used by the layer call wrapper.
  # Inputs: layer (unused), inputs, args, kwargs (call arguments) and input_list (unused). Returns
  # true if any tensor in the nested inputs, args or kwargs is a symbolic keras tensor, meaning a
  # functional model is being constructed.
  """Check the arguments to see if we are constructing a functional model."""
  # We are constructing a functional model if any of the inputs
  # are KerasTensors
  return any(
      isinstance(tensor, keras_tensor.KerasTensor)
      for tensor in nest.flatten([inputs, args, kwargs]))


def _convert_numpy_or_python_types(x):
  # [FUNCTION DOC]
  # HELPER: Mapping callback of the layer call wrapper. Converts numpy arrays, Python floats and
  # ints to tensors and returns any other value unchanged.
  if isinstance(x, (np_arrays.ndarray, np.ndarray, float, int)):
    return tensor_conversion.convert_to_tensor_v2_with_dispatch(x)
  return x


# Avoid breaking users who directly import this symbol from this file.
# TODO(fchollet): remove this.
InputSpec = input_spec.InputSpec  # pylint:disable=invalid-name
