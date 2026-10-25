# [MODULE DOC]
# Companion configuration classes for the TPU embedding mid level API (TF2). It defines the
# optimizer parameter classes (SGD, Adagrad, AdagradMomentum, FTRL, Adam and the user-defined
# CustomOptimizer), the CustomCombiner, QuantizationConfig, TableConfig, FeatureConfig and the
# RowIdInitializer, which are exported under the public TPU experimental embedding namespace.
# Each optimizer knows its slot variable names and initializers, which TPU load/retrieve
# operation to use, and how to fill the optimization parameters protocol buffer. TableConfig
# knows how to fill a table descriptor of the TPU embedding configuration. It also provides
# helpers to log a large configuration in pieces and to list the hosts of a TPU strategy in a
# deterministic sorted order. Import time side effects: only the public-name export
# registrations; no flags or global state are created.
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
"""Companion classes for mid level API for TPU Embeddings in TF2."""

import abc
import math
import typing
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Text, Tuple, TypeVar, Union

from absl import logging

from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export


TableVariable = TypeVar("TableVariable", sharded_variable.ShardedVariable,
                        tf_variables.Variable)
SlotVarCreationFnType = Callable[
    [TableVariable, List[Text], List[init_ops_v2.Initializer]],
    Dict[Text, TableVariable]]
ClipValueType = Union[Tuple[float, float], float]


class _WithSlotVariables(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # HELPER: abstract base shared by the optimizers and the custom combiner of this module.
  # Abstract base for objects that may need auxiliary ("slot") variables next to an embedding
  # table. It stores an optional user callback that creates slot variables, declares the
  # abstract slot name and slot initializer lists, and implements the generic slot creation
  # plus attribute based equality and hashing. Subclasses supply the slot names/initializers.
  """Base class that allows slot variables to be created."""

  def __init__(
      self,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
  ):
    # [FUNCTION DOC]
    # Stores the optional slot variable creation callback. Input: slot_variable_creation_fn,
    # either nothing (use built-in creation) or a callable taking a table, a list of slot
    # names and a list of initializers and returning a mapping of slot name to variable.
    # Returns nothing. Raises a value error if the argument is neither nothing nor callable.
    # Side effect: sets the instance attribute holding the callback.
    if slot_variable_creation_fn is not None and not callable(
        slot_variable_creation_fn
    ):
      raise ValueError(
          "Argument `slot_variable_creation_fn` must be either None or a "
          f"callable. Received: {slot_variable_creation_fn}"
      )
    self.slot_variable_creation_fn = slot_variable_creation_fn

  @abc.abstractmethod
  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # HELPER: abstract hook used by slot variable creation.
    # Abstract: subclasses return the list of slot variable names (excluding the main
    # parameters variable). The base implementation raises a not-implemented error.
    """Returns the name of all the slot variables.

    This does not include the 'parameters' variable and these names must match
    the names of the slots variables as used in the corresponding
    `tpu_ops.load_tpu_embedding_*` ops.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # HELPER: abstract hook used by slot variable creation.
    # Abstract: subclasses return a list of initializers parallel to the slot names. The base
    # implementation raises a not-implemented error.
    """Returns initializers for slot variables.

    This returns a parallel list to self._slot_names().
    """
    raise NotImplementedError

  def _create_slots(
      self,
      table: "TableConfig",
      variable_creator: Callable[
          [Text, init_ops_v2.Initializer], tf_variables.Variable
      ],
      initializer_wrapper: Optional[
          Callable[[str, init_ops_v2.Initializer], init_ops_v2.Initializer]
      ] = None,
  ) -> Dict[Text, tf_variables.Variable]:
    # [FUNCTION DOC]
    # HELPER: internal glue used when the embedding layer builds slot variables for a table.
    # Inputs: table, the table (or table variable) to create slots for; variable_creator, a
    # callable taking a slot name and an initializer and returning a new variable;
    # initializer_wrapper, optional callable taking a slot name and initializer and returning
    # a replacement initializer. Output: mapping from slot name to variable. If a custom slot
    # creation callback was supplied at construction, it is called with the table, names and
    # (wrapped) initializers and its result is returned; otherwise each slot is created with
    # the variable creator. Side effects: whatever the creator callbacks do (variable creation).
    """Creates slot variables for table.

    Args:
      table: The table variable to create slots for.
      variable_creator: A function which creates variables. Takes parameters
        'name', 'initializer'.
      initializer_wrapper: A function that wraps the initializer.

    Returns:
      A dict of variables, keyed by self._slot_names().
    """
    names = self._slot_names()
    initializers = self._slot_initializers()

    if initializer_wrapper is not None:
      initializers = [
          initializer_wrapper(name, initializer)
          for name, initializer in zip(names, initializers)
      ]

    if self.slot_variable_creation_fn is not None:
      return self.slot_variable_creation_fn(table, names, initializers)
    else:
      slots = {}
      for slot, initializer in zip(names, initializers):
        slots[slot] = variable_creator(slot, initializer)
      return slots

  def __eq__(self, other: Any) -> Union[Any, bool]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: compares attributes pairwise by position in the attribute mapping, so it
    # relies on identical insertion order and does not check that both have the same number of
    # attributes (zip stops at the shorter one).
    # Equality: true when other is an instance of this object's class and all attribute
    # (name, value) pairs match in order; false for any other type. No side effects.
    if isinstance(other, self.__class__):
      return all([
          attr1 == attr2
          for attr1, attr2 in zip(self.__dict__.items(), other.__dict__.items())
      ])
    else:
      return False

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Returns an integer hash built from all instance attribute (name, value) pairs, so it is
    # consistent with equality. Raises a type error if any attribute value is unhashable.
    return hash(tuple(self.__dict__.items()))


class _Optimizer(_WithSlotVariables):
  # [CLASS DOC]
  # HELPER: common base for all TPU embedding optimizer parameter classes.
  # Holds the parameters shared by every optimizer: learning rate (constant or callable),
  # gradient accumulation flag, weight clipping limits, gradient clipping limits, weight decay
  # settings and the low dimensional packing flag. Provides filling of the common part of the
  # optimization parameters protocol buffer and declares abstract load/retrieve hooks that
  # return the matching TPU operations.
  """Base class for all optimizers, with common parameters."""

  def __init__(
      self,
      learning_rate: Union[float, Callable[[], float]],
      use_gradient_accumulation: bool,
      clip_weight_min: Optional[float],
      clip_weight_max: Optional[float],
      weight_decay_factor: Optional[float],
      multiply_weight_decay_factor_by_learning_rate: bool,
      clipvalue: Optional[ClipValueType] = None,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
      low_dimensional_packing_status: bool = False,
  ):
    # [FUNCTION DOC]
    # HELPER: shared constructor body for the concrete optimizers.
    # Stores the common settings. Inputs: learning_rate (float or no-argument callable for a
    # dynamic rate); use_gradient_accumulation flag; clip_weight_min/max (optional weight
    # limits); weight_decay_factor (optional); multiply_weight_decay_factor_by_learning_rate
    # flag; clipvalue (optional single positive number meaning symmetric limits, or a pair of
    # minimum and maximum where either may be nothing); slot_variable_creation_fn (optional);
    # low_dimensional_packing_status flag. Gradient clip limits are derived from clipvalue
    # (both nothing when unset). Raises a value error if clipvalue is given while gradient
    # accumulation is disabled. Returns nothing; sets instance attributes.
    super().__init__(slot_variable_creation_fn=slot_variable_creation_fn)
    self.learning_rate = learning_rate
    self.use_gradient_accumulation = use_gradient_accumulation
    self.clip_weight_min = clip_weight_min
    self.clip_weight_max = clip_weight_max
    if not use_gradient_accumulation and clipvalue is not None:
      raise ValueError(
          "When `use_gradient_accumulation` is False, gradient clipping "
          "cannot be used and `clipvalue` should be left as None. "
          f"Received value {clipvalue} for argument `clipvalue`."
      )
    if clipvalue is None:
      clipvalue = (None, None)
    elif not isinstance(clipvalue, tuple):
      clipvalue = (-1.0 * clipvalue, clipvalue)
    self.clip_gradient_min, self.clip_gradient_max = clipvalue

    self.weight_decay_factor = weight_decay_factor
    self.multiply_weight_decay_factor_by_learning_rate = (
        multiply_weight_decay_factor_by_learning_rate
    )

    self.low_dimensional_packing_status = low_dimensional_packing_status

  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters
  ):
    # [FUNCTION DOC]
    # HELPER: fills the common part of the optimization parameters message for a table.
    # Input: parameters, the optimization parameters protocol buffer, which is mutated: sets
    # gradient accumulation enabled/disabled, weight clipping limits and gradient clipping
    # limits when not nothing, the weight decay factor (and the multiply-by-learning-rate flag)
    # when the factor is non-zero, and the low dimensional packing status. Returns nothing.
    """Sets the optimizer fields in the OptimizationParameters."""
    if self.use_gradient_accumulation:
      parameters.gradient_accumulation_status = (
          optimization_parameters_pb2.GradientAccumulationStatus.ENABLED
      )
    else:
      parameters.gradient_accumulation_status = (
          optimization_parameters_pb2.GradientAccumulationStatus.DISABLED
      )

    if self.clip_weight_min is not None:
      parameters.clipping_limits.lower.value = self.clip_weight_min

    if self.clip_weight_max is not None:
      parameters.clipping_limits.upper.value = self.clip_weight_max

    if self.clip_gradient_min is not None:
      parameters.gradient_clipping_limits.lower.value = self.clip_gradient_min

    if self.clip_gradient_max is not None:
      parameters.gradient_clipping_limits.upper.value = self.clip_gradient_max

    if self.weight_decay_factor:
      parameters.weight_decay_factor = self.weight_decay_factor
      if self.multiply_weight_decay_factor_by_learning_rate:
        parameters.multiply_weight_decay_factor_by_learning_rate = True

    parameters.low_dimensional_packing_status = (
        self.low_dimensional_packing_status
    )

  @abc.abstractmethod
  def _load(self) -> Callable[..., ops.Operation]:
    # [FUNCTION DOC]
    # HELPER: abstract hook; subclasses return the TPU operation used to load table
    # parameters for this optimizer. The base implementation raises a not-implemented error.
    """Returns the load function for the optimizer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _retrieve(self) -> Callable[..., core.Tensor]:
    # [FUNCTION DOC]
    # HELPER: abstract hook; subclasses return the TPU operation used to retrieve table
    # parameters for this optimizer. The base implementation raises a not-implemented error.
    """Returns the retrieve function for the optimizer."""
    raise NotImplementedError


@tf_export("tpu.experimental.embedding.CustomOptimizer")
class CustomOptimizer(_Optimizer):
  # [CLASS DOC]
  # Optimizer whose update rule is a user supplied per-row computation (a traced function),
  # intended only for the V2 TPU embedding class on newer TPUs. It stores the computation,
  # learning rate, hyperparameters, slot names, slot initializers and optional slot dimensions
  # (1 or 2 each). Gradient accumulation, clipping and weight decay are fixed to off. It does not
  # support the load and retrieve operations. Created by users and passed as a global or
  # per-table optimizer; exported publicly.
  """Optimization parameters for custom optimizer for TPU embeddings.

  This optimizer gives the user the ability to define a custom optimizer
  for running embedding lookups on TPU v5p.

  The custom computation should be a function which takes gradient, embedding
  table, a list of slot variables, learning_rate and a list of hyperparameters.
  The function should perform the gradient update on the embedding_table +
  slot_variables and return the updated embedding_table and slot_variables. e.g.

  ```python
  @tf.function
  def sgd_optimizer_computation(
      gradient,
      embedding_table,
      slot_variables,
      learning_rate,
      hyperparameters,
  ):
    del slot_variables, hyperparameters
    return embedding_table - gradient * learning_rate
  ```

  Above is a simple example of a sgd optimizer. You can also define a more
  complex optimizer which updates multiple tables and slot variables.

  ```python
  @def_function.function
  def adagrad_optimizer_computation(
      gradient,
      embedding_table,
      slot_variables,
      learning_rate,
      hyperparameters,
  ):
    del hyperparameters
    accumulator = slot_variables[0]
    new_accumulator = accumulator + gradient * gradient
    updated_embedding_table = (
        embedding_table
        - learning_rate * gradient / math_ops.sqrt(new_accumulator)
    )
    return (updated_embedding_table, new_accumulator)
  ```

  The custom computation is defined as a per-row update function and it will be
  auto scaled for the entire table (slot variables).

  NOTE: This optimizer can only be used with the `TPUEmbeddingV2` class.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbeddingV2` via the
  `optimizer` argument to set the global optimizer and its parameters:

  ```python
  optimizer = tf.tpu.experimental.embedding.CustomOptimizer(
        custom_computation=adagrad_optimizer_computation,
        learning_rate=1.0,
        slot_names=['accumulators'],
        slot_initializers=[
            tf.constant_initializer(0.1, support_partition=True)
        ],
    )
  embedding = tf.tpu.experimental.embedding.TPUEmbeddingV2(
      ...
      optimizer=optimizer)
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=optimizer)
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=optimizer)
  ```
  In this example, the optimizer of the first table will be the one specified
  in the table config. The second table will use the optimizer specified in the
  TPUEmbedding argument.
  """

  def __init__(
      self,
      custom_computation: core.PolymorphicFunction,
      learning_rate: Union[float, Callable[[], float]] = 0.01,
      slot_names: Optional[List[str]] = None,
      slot_initializers: Optional[List[init_ops_v2.Initializer]] = None,
      hyperparameters: Optional[List[Union[float, Callable[[], float]]]] = None,
      slot_dims: Optional[List[int]] = None,
  ) -> Any:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the constructor is annotated as returning an arbitrary value, but like any
    # constructor it returns nothing; also it passes nothing for the multiply-weight-decay flag
    # to the base class, which declares it as a boolean.
    # Inputs: custom_computation, the traced function performing the per-row update (gradient,
    # table row, slot variables, learning rate, hyperparameters); learning_rate (float or
    # callable, default 0.01); slot_names (optional list of strings); slot_initializers
    # (optional list, must match slot_names in length); hyperparameters (optional list of
    # numbers or callables); slot_dims (optional list of 1 or 2 per slot, same length as
    # slot_names when given). Raises a value error when lengths mismatch or a slot dimension
    # is not 1 or 2. Stores everything as immutable tuples.
    super().__init__(  # pytype: disable=wrong-arg-types
        learning_rate,
        use_gradient_accumulation=False,
        clip_weight_min=None,
        clip_weight_max=None,
        weight_decay_factor=None,
        multiply_weight_decay_factor_by_learning_rate=None,
        clipvalue=None,
        slot_variable_creation_fn=None,
        low_dimensional_packing_status=False,
    )
    self._slot_names_attr = tuple(slot_names if slot_names else ())
    self._slot_initializers_attr = tuple(
        slot_initializers if slot_initializers else ()
    )
    self._slot_dims_attr = tuple(slot_dims if slot_dims else ())
    num_slot_names = len(self._slot_names_attr)
    num_slot_initializers = len(self._slot_initializers_attr)
    if num_slot_names != num_slot_initializers:
      raise ValueError(
          f"The number of slot_names ({num_slot_names}) must match"
          " the number of slot_initializers"
          f" ({num_slot_initializers})."
      )
    if self._slot_dims_attr:
      if len(self._slot_dims_attr) != num_slot_names:
        raise ValueError(
            f"The number of slot_dims ({len(self._slot_dims_attr)}) must match "
            f"the number of slot_names ({num_slot_names})."
        )
      for i, dim in enumerate(self._slot_dims_attr):
        if dim not in (1, 2):
          raise ValueError(
              f"Unsupported slot dim '{dim}' for slot"
              f" '{self._slot_names_attr[i]}'. Only 1D and 2D are supported."
          )
    self._hyperparameters = tuple(hyperparameters if hyperparameters else ())
    self._custom_computation = custom_computation

  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # Returns a new list of the user supplied slot names (possibly empty).
    return list(self._slot_names_attr)

  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # Returns a new list of the user supplied slot initializers, parallel to the slot names.
    return list(self._slot_initializers_attr)

  def _load(self) -> Callable[..., ops.Operation]:
    # [FUNCTION DOC]
    # Always raises a not-implemented error: this optimizer is only usable with the V2 class.
    raise NotImplementedError(
        "Custom optimizer does not support load op since it is only used for"
        " TPUEmbeddingV2."
    )

  def _retrieve(self) -> Callable[..., core.Tensor]:
    # [FUNCTION DOC]
    # Always raises a not-implemented error: this optimizer is only usable with the V2 class.
    raise NotImplementedError(
        "Custom optimizer does not support retrieve op since it is only used"
        " for TPUEmbeddingV2."
    )

  @property
  def hyperparameters(self) -> Tuple[Union[float, Callable[[], float]], ...]:
    # [FUNCTION DOC]
    # Read-only property: the tuple of hyperparameters (numbers or callables) given at
    # construction, empty if none.
    return self._hyperparameters

  @property
  def custom_computation(self) -> core.ConcreteFunction:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as a concrete function, but the stored object is whatever
    # polymorphic function the caller passed to the constructor (no conversion happens).
    # Read-only property: returns the user supplied update computation.
    return self._custom_computation

  @property
  def slot_dims(self) -> Tuple[int, ...]:
    # [FUNCTION DOC]
    # Read-only property: tuple of per-slot dimensionalities (each 1 or 2); empty when none were
    # given. Consumed by the embedding layer when creating slot variables.
    """Optional per-slot dimensions metadata for variable creation.

    Each entry corresponds to the same-index item in `slot_names`. This is
    consumed by the embedding layer when creating slot variables.
    """
    return self._slot_dims_attr


@tf_export("tpu.experimental.embedding.SGD")
class SGD(_Optimizer):
  # [CLASS DOC]
  # Stochastic gradient descent optimizer parameters for TPU embedding tables. Has no slot
  # variables; selects the SGD section of the optimization parameters message and the SGD
  # load/retrieve TPU operations. Used as a global or per-table optimizer; exported publicly.
  """Optimization parameters for stochastic gradient descent for TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.SGD(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(
      self,
      learning_rate: Union[float, Callable[[], float]] = 0.01,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clipvalue: Optional[ClipValueType] = None,
      low_dimensional_packing_status: bool = False,
  ):
    # [FUNCTION DOC]
    # Creates SGD parameters. Inputs: learning_rate (float or no-argument callable, default
    # 0.01); use_gradient_accumulation (default true); optional weight clip minimum/maximum;
    # optional weight decay factor and its multiply-by-learning-rate flag; optional clipvalue
    # (single value or minimum/maximum pair; requires gradient accumulation); and the low
    # dimensional packing flag. Delegates to the common optimizer constructor (no slot creation
    # callback). Returns nothing; raises a value error if clipvalue is used without
    # gradient accumulation.
    """Optimization parameters for stochastic gradient descent.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed. Weights are decayed by multiplying the weight
        by this factor each step.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tiple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction. Note if this is
        set, you may see a decrease in performance as  gradient accumulation
        will be enabled (it is normally off for SGD as it has no affect on
        accuracy). See
        'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for more
        information on gradient accumulation and its impact on tpu embeddings.
      low_dimensional_packing_status: Status of the low-dimensional embedding
        packing optimization controls whether to optimize the packing of
        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in
        memory.
    """
    super().__init__(
        learning_rate,
        use_gradient_accumulation,
        clip_weight_min,
        clip_weight_max,
        weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        clipvalue,
        None,
        low_dimensional_packing_status,
    )

  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # Returns an empty list: SGD needs no slot variables.
    return []

  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # Returns an empty list: SGD needs no slot variables.
    return []

  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters):
    # [FUNCTION DOC]
    # Fills the common optimization fields and marks the stochastic gradient descent section
    # as present on the given parameters message (mutated in place). Returns nothing.
    super()._set_optimization_parameters(parameters)
    parameters.stochastic_gradient_descent.SetInParent()

  def _load(self) -> Callable[..., ops.Operation]:
    # [FUNCTION DOC]
    # Returns the TPU operation that loads SGD embedding table parameters.
    return tpu_ops.load_tpu_embedding_stochastic_gradient_descent_parameters

  def _retrieve(self) -> Callable[..., core.Tensor]:
    # [FUNCTION DOC]
    # Returns the TPU operation that retrieves SGD embedding table parameters.
    return tpu_ops.retrieve_tpu_embedding_stochastic_gradient_descent_parameters


@tf_export("tpu.experimental.embedding.Adagrad")
class Adagrad(_Optimizer):
  # [CLASS DOC]
  # Adagrad optimizer parameters for TPU embedding tables. Keeps one slot variable
  # (accumulators) initialized to a positive constant. Selects the Adagrad section of the
  # optimization parameters message and the Adagrad load/retrieve TPU operations. Used as a
  # global or per-table optimizer; exported publicly.
  """Optimization parameters for Adagrad with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(
      self,
      learning_rate: Union[float, Callable[[], float]] = 0.001,
      initial_accumulator_value: float = 0.1,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
      clipvalue: Optional[ClipValueType] = None,
      low_dimensional_packing_status: bool = False,
  ):
    # [FUNCTION DOC]
    # Creates Adagrad parameters. Inputs: learning_rate (float or no-argument callable,
    # default 0.001); initial_accumulator_value (must be positive, default 0.1);
    # gradient accumulation flag (default true); optional weight clip limits; optional weight
    # decay factor and multiply-by-learning-rate flag; optional slot creation callback;
    # optional clipvalue; low dimensional packing flag. Delegates common settings to the base
    # constructor. Raises a value error if the initial accumulator value is not positive (or
    # if clipvalue is used without gradient accumulation). Returns nothing.
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      initial_accumulator_value: initial accumulator for Adagrad.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
        table variable, a list of slot names to create for it, and a list of
        initializers. This function should return a dict with the slot names as
        keys and the created variables as values with types matching the table
        variable. When set to None (the default), uses the built-in variable
        creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tuple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
      low_dimensional_packing_status: Status of the low-dimensional embedding
        packing optimization controls whether to optimize the packing of
        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in
        memory.
    """
    super().__init__(
        learning_rate,
        use_gradient_accumulation,
        clip_weight_min,
        clip_weight_max,
        weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        clipvalue,
        slot_variable_creation_fn,
        low_dimensional_packing_status,
    )
    if initial_accumulator_value <= 0:
      raise ValueError(
          f"Argument `initial_accumulator_value` must be a positive float. "
          f"Received: {initial_accumulator_value}")
    self.initial_accumulator_value = initial_accumulator_value

  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # Returns the list with the single slot name "accumulators".
    return ["accumulators"]

  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # Returns a one element list: a constant initializer (partition aware) set to the initial
    # accumulator value.
    return [
        init_ops_v2.Constant(
            self.initial_accumulator_value, support_partition=True
        )
    ]

  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters):
    # [FUNCTION DOC]
    # Fills the common optimization fields and marks the Adagrad section as present on the
    # given parameters message (mutated in place). Returns nothing.
    super()._set_optimization_parameters(parameters)
    parameters.adagrad.SetInParent()

  def _load(self) -> Callable[..., ops.Operation]:
    # [FUNCTION DOC]
    # Returns the TPU operation that loads Adagrad embedding table parameters.
    return tpu_ops.load_tpu_embedding_adagrad_parameters

  def _retrieve(self) -> Callable[..., core.Tensor]:
    # [FUNCTION DOC]
    # Returns the TPU operation that retrieves Adagrad embedding table parameters.
    return tpu_ops.retrieve_tpu_embedding_adagrad_parameters


@tf_export("tpu.experimental.embedding.AdagradMomentum")
class AdagradMomentum(_Optimizer):
  # [CLASS DOC]
  # Adagrad with momentum optimizer parameters for TPU embedding tables. Keeps two slot
  # variables (accumulators and momenta), both zero initialized. Stores momentum, Nesterov
  # flag, accumulator exponent, beta2 and epsilon, and writes them to the Adagrad-momentum
  # section of the optimization parameters message. Exported publicly.
  """Optimization parameters for Adagrad + Momentum with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.AdagradMomentum(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(
      self,
      learning_rate: Union[float, Callable[[], float]] = 0.001,
      momentum: float = 0.0,
      use_nesterov: bool = False,
      exponent: float = 2,
      beta2: float = 1,
      epsilon: float = 1e-10,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
      clipvalue: Optional[ClipValueType] = None,
      low_dimensional_packing_status: bool = False,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring describes epsilon as the initial accumulator, but it is
    # written to the epsilon field of the parameters message and only validated to be positive.
    # Creates Adagrad+momentum parameters. Inputs: learning_rate (default 0.001); momentum
    # (default 0); use_nesterov flag; exponent (default 2, must be positive); beta2 (default 1);
    # epsilon (default 1e-10, must be positive); plus the common optimizer settings (gradient
    # accumulation, weight clipping, weight decay, slot creation callback, clipvalue, low
    # dimensional packing). Raises a value error for non-positive epsilon or exponent.
    # Returns nothing.
    """Optimization parameters for Adagrad + Momentum.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      momentum: Moving average parameter for the momentum accumulator.
      use_nesterov: Whether to use the Nesterov variant of momentum. See
        Sutskever et al., 2013.
      exponent: Exponent for the Adagrad accumulator.
      beta2: Moving average parameter for the Adagrad accumulator.
      epsilon: initial accumulator for Adagrad accumulator.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
        table variable, a list of slot names to create for it, and a list of
        initializers. This function should return a dict with the slot names as
        keys and the created variables as values with types matching the table
        variable. When set to None (the default), uses the built-in variable
        creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tuple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
      low_dimensional_packing_status: Status of the low-dimensional embedding
        packing optimization controls whether to optimize the packing of
        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in
        memory.
    """
    super().__init__(
        learning_rate,
        use_gradient_accumulation,
        clip_weight_min,
        clip_weight_max,
        weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        clipvalue,
        slot_variable_creation_fn,
        low_dimensional_packing_status,
    )
    if epsilon <= 0:
      raise ValueError("Adagrad momentum: epsilon must be positive")
    if exponent <= 0:
      raise ValueError("Adagrad momentum: Precondition exponent must >0")
    self.momentum = momentum
    self.use_nesterov = use_nesterov
    self.exponent = exponent
    self.beta2 = beta2
    self.epsilon = epsilon

  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # Returns the slot names "accumulators" and "momenta".
    return ["accumulators", "momenta"]

  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # Returns two default (zero valued) partition aware constant initializers, one per slot.
    return [
        init_ops_v2.Constant(support_partition=True),
        init_ops_v2.Constant(support_partition=True),
    ]

  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters
  ):
    # [FUNCTION DOC]
    # Fills the common optimization fields, then marks the Adagrad-momentum section present and
    # writes momentum, Nesterov flag, exponent, beta2 and epsilon into the given parameters
    # message (mutated in place). Returns nothing.
    super()._set_optimization_parameters(parameters)
    parameters.adagrad_momentum.SetInParent()
    parameters.adagrad_momentum.momentum = self.momentum
    parameters.adagrad_momentum.use_nesterov = self.use_nesterov
    parameters.adagrad_momentum.exponent = self.exponent
    parameters.adagrad_momentum.beta2 = self.beta2
    parameters.adagrad_momentum.epsilon = self.epsilon

  def _load(self) -> Callable[..., ops.Operation]:
    # [FUNCTION DOC]
    # Returns the TPU operation that loads Adagrad-momentum embedding table parameters.
    return tpu_ops.load_tpu_embedding_adagrad_momentum_parameters

  def _retrieve(self) -> Callable[..., core.Tensor]:
    # [FUNCTION DOC]
    # Returns the TPU operation that retrieves Adagrad-momentum embedding table parameters.
    return tpu_ops.retrieve_tpu_embedding_adagrad_momentum_parameters


@tf_export("tpu.experimental.embedding.FTRL")
class FTRL(_Optimizer):
  # [CLASS DOC]
  # Follow-the-regularized-leader optimizer parameters for TPU embedding tables. Keeps two
  # slot variables (accumulators, initialized to a positive constant, and linears, zero
  # initialized). Stores learning rate power, L1/L2 strengths, beta and two behavior flags, and
  # writes them to the FTRL section of the optimization parameters message. Exported publicly.
  """Optimization parameters for FTRL with TPU embeddings.

  See Algorithm 1 of this
  [paper](https://research.google.com/pubs/archive/41159.pdf).

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.FTRL(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(
      self,
      learning_rate: Union[float, Callable[[], float]] = 0.001,
      learning_rate_power: float = -0.5,
      l1_regularization_strength: float = 0.0,
      l2_regularization_strength: float = 0.0,
      beta: float = 0.0,
      initial_accumulator_value: float = 0.1,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
      clipvalue: Optional[ClipValueType] = None,
      multiply_linear_by_learning_rate: bool = False,
      allow_zero_accumulator: bool = False,
      low_dimensional_packing_status: bool = False,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it configures Adagrad (copy-paste; it configures FTRL),
    # and says zero initial accumulator values are allowed, but the code raises an error for
    # any value that is not strictly positive. Range checks promised for learning_rate_power
    # and the L1/L2 strengths are not enforced.
    # Creates FTRL parameters. Inputs: learning_rate (default 0.001); learning_rate_power
    # (default -0.5); l1 and l2 regularization strengths (default 0); beta (default 0);
    # initial_accumulator_value (default 0.1); plus the common optimizer settings (gradient
    # accumulation, weight clipping, weight decay, slot creation callback, clipvalue);
    # multiply_linear_by_learning_rate and allow_zero_accumulator flags; low dimensional
    # packing flag. Raises a value error for a non-positive initial accumulator. Returns nothing.
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for a
        fixed learning rate.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      beta: A float value, representing the beta value from the paper.
      initial_accumulator_value: The starting value for accumulators. Only zero
        or positive values are allowed.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
        table variable, a list of slot names to create for it, and a list of
        initializers. This function should return a dict with the slot names as
        keys and the created variables as values with types matching the table
        variable. When set to None (the default), uses the built-in variable
        creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tuple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
      multiply_linear_by_learning_rate: If set to True, a modified formula is
        used for FTRL that treats the "linear" accumulator as being
        pre-multiplied by the learning rate (i.e., the accumulator named
        "linear" actually stores "linear * learning_rate"). Other than
        checkpoint compatibility, this is mathematically equivalent for a static
        learning rate; for a dynamic learning rate, it is nearly the same as
        long as the learning rate does not change quickly. The benefit of this
        is that the modified formula handles zero and near-zero learning rates
        without producing NaNs, improving flexibility for learning rate ramp-up.
      allow_zero_accumulator: If set to True, changes some internal formulas to
        allow zero and near-zero accumulator values at the cost of some
        performance; this only needs to be set if you are using an initial
        accumulator value of zero, which is uncommon.
      low_dimensional_packing_status: Status of the low-dimensional embedding
        packing optimization controls whether to optimize the packing of
        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in
        memory.
    """
    super().__init__(
        learning_rate,
        use_gradient_accumulation,
        clip_weight_min,
        clip_weight_max,
        weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        clipvalue,
        slot_variable_creation_fn,
        low_dimensional_packing_status,
    )
    if initial_accumulator_value <= 0:
      raise ValueError(
          f"Argument `initial_accumulator_value` must be a positive float. "
          f"Received: {initial_accumulator_value}")
    self.initial_accumulator_value = initial_accumulator_value
    self.learning_rate_power = learning_rate_power
    self.l1_regularization_strength = l1_regularization_strength
    self.l2_regularization_strength = l2_regularization_strength
    self.beta = beta
    self.multiply_linear_by_learning_rate = multiply_linear_by_learning_rate
    self.allow_zero_accumulator = allow_zero_accumulator

  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # Returns the slot names "accumulators" and "linears".
    return ["accumulators", "linears"]

  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # Returns two partition aware constant initializers: the accumulators start at the initial
    # accumulator value, the linears at zero.
    return [
        init_ops_v2.Constant(
            self.initial_accumulator_value, support_partition=True
        ),
        init_ops_v2.Constant(support_partition=True),
    ]

  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters
  ):
    # [FUNCTION DOC]
    # Fills the common optimization fields, then writes L1, L2, learning rate power, beta, the
    # multiply-linear-by-learning-rate flag and the allow-zero-accumulator flag into the FTRL
    # section of the given parameters message (mutated in place). Returns nothing.
    super()._set_optimization_parameters(parameters)
    ftrl = parameters.ftrl
    ftrl.l1 = self.l1_regularization_strength
    ftrl.l2 = self.l2_regularization_strength
    ftrl.lr_power = self.learning_rate_power
    ftrl.beta = self.beta
    ftrl.multiply_linear_by_lr = self.multiply_linear_by_learning_rate
    ftrl.allow_zero_accumulator = self.allow_zero_accumulator

  def _load(self) -> Callable[..., ops.Operation]:
    # [FUNCTION DOC]
    # Returns the TPU operation that loads FTRL embedding table parameters.
    return tpu_ops.load_tpu_embedding_ftrl_parameters

  def _retrieve(self) -> Callable[..., core.Tensor]:
    # [FUNCTION DOC]
    # Returns the TPU operation that retrieves FTRL embedding table parameters.
    return tpu_ops.retrieve_tpu_embedding_ftrl_parameters


@tf_export("tpu.experimental.embedding.Adam")
class Adam(_Optimizer):
  # [CLASS DOC]
  # Adam optimizer parameters for TPU embedding tables (lazy Adam by default). Keeps two slot
  # variables (momenta and velocities), both zero initialized. Stores beta_1, beta_2, epsilon,
  # the lazy flag and the sum-inside-square-root flag, and writes them to the Adam section of
  # the optimization parameters message. Exported publicly.
  """Optimization parameters for Adam with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  NOTE: By default this optimizer is lazy, i.e. it will not apply the gradient
  update of zero to rows that were not looked up. You can change this behavior
  by setting `lazy_adam` to `False`.

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adam(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(
      self,
      learning_rate: Union[float, Callable[[], float]] = 0.001,
      beta_1: float = 0.9,
      beta_2: float = 0.999,
      epsilon: float = 1e-07,
      lazy_adam: bool = True,
      sum_inside_sqrt: bool = True,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
      clipvalue: Optional[ClipValueType] = None,
      low_dimensional_packing_status: bool = False,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error text for an out-of-range beta_2 reports the value of beta_1, and
    # the error text for non-lazy Adam without accumulation tells the user to set gradient
    # accumulation to False although it must be enabled.
    # Creates Adam parameters. Inputs: learning_rate (default 0.001); beta_1 and beta_2 (each
    # must be in zero inclusive to one exclusive; defaults 0.9 and 0.999); epsilon (positive,
    # default 1e-7); lazy_adam flag (default true); sum_inside_sqrt flag (default true); plus
    # the common optimizer settings. Raises a value error for out-of-range beta values,
    # non-positive epsilon, or when lazy Adam is off while gradient accumulation is off.
    # Returns nothing.
    """Optimization parameters for Adam.

    See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
    complete description of these parameters and their impacts on the optimizer
    algorithm.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      beta_1: A float value. The exponential decay rate for the 1st moment
        estimates.
      beta_2: A float value. The exponential decay rate for the 2nd moment
        estimates.
      epsilon: A small constant for numerical stability.
      lazy_adam: Use lazy Adam instead of Adam. Lazy Adam trains faster.
      sum_inside_sqrt: When this is true, the Adam update formula is changed
        from `m / (sqrt(v) + epsilon)` to `m / sqrt(v + epsilon**2)`. This
        option improves the performance of TPU training and is not expected to
        harm model quality.
      use_gradient_accumulation: Setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
        table variable, a list of slot names to create for it, and a list of
        initializers. This function should return a dict with the slot names as
        keys and the created variables as values with types matching the table
        variable. When set to None (the default), uses the built-in variable
        creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tiple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
      low_dimensional_packing_status: Status of the low-dimensional embedding
        packing optimization controls whether to optimize the packing of
        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in
        memory.
    """
    super(Adam, self).__init__(
        learning_rate,
        use_gradient_accumulation,
        clip_weight_min,
        clip_weight_max,
        weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        clipvalue,
        slot_variable_creation_fn,
        low_dimensional_packing_status,
    )
    if beta_1 < 0. or beta_1 >= 1.:
      raise ValueError(
          f"Argument `beta_1` must be >= 0 and < 1. Received: {beta_1}.")
    if beta_2 < 0. or beta_2 >= 1.:
      raise ValueError(
          f"Argument `beta_2` must be >= 0 and < 1. Received: {beta_1}.")
    if epsilon <= 0.:
      raise ValueError("epsilon must be positive; got {}.".format(epsilon))
    if not use_gradient_accumulation and not lazy_adam:
      raise ValueError(
          "When disabling lazy Adam (`lazy_adam=False`), "
          "gradient accumulation must be used. "
          "Set `use_gradient_accumulation` to False.")

    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.lazy_adam = lazy_adam
    self.sum_inside_sqrt = sum_inside_sqrt

  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # Returns the slot names "momenta" and "velocities".
    return ["momenta", "velocities"]

  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # Returns two default (zero valued) partition aware constant initializers, one per slot.
    return [
        init_ops_v2.Constant(support_partition=True),
        init_ops_v2.Constant(support_partition=True),
    ]

  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters
  ):
    # [FUNCTION DOC]
    # Fills the common optimization fields, then writes beta1, beta2, epsilon, the non-lazy
    # flag (the opposite of lazy_adam) and the sum-inside-square-root flag into the Adam
    # section of the given parameters message (mutated in place). Returns nothing.
    super(Adam, self)._set_optimization_parameters(parameters)
    parameters.adam.beta1 = self.beta_1
    parameters.adam.beta2 = self.beta_2
    parameters.adam.epsilon = self.epsilon
    parameters.adam.use_non_lazy_adam = not self.lazy_adam
    parameters.adam.use_sum_inside_sqrt = self.sum_inside_sqrt

  def _load(self) -> Callable[..., ops.Operation]:
    # [FUNCTION DOC]
    # Returns the TPU operation that loads Adam embedding table parameters.
    return tpu_ops.load_tpu_embedding_adam_parameters

  def _retrieve(self) -> Callable[..., core.Tensor]:
    # [FUNCTION DOC]
    # Returns the TPU operation that retrieves Adam embedding table parameters.
    return tpu_ops.retrieve_tpu_embedding_adam_parameters


@tf_export("tpu.experimental.embedding.CustomCombiner")
class CustomCombiner(_WithSlotVariables):
  # [CLASS DOC]
  # User defined combiner for embedding lookups on SparseCore (V2 class only). Wraps a traced
  # per-sample combiner function with its maximum valency, optional number of trainable combiner
  # weights, an initializer for those weights and their learning rate. Exposes a generated
  # name starting with a fixed prefix, and one slot variable (for the weights) only when the
  # number of weights is positive. Used as the combiner of a table configuration.
  """Custom combiner for TPU embeddings.

  This class gives the user the ability to define a custom combiner for running
  embedding lookups on TPU with SparseCores.

  The custom computation should be a function which takes the following
  arguments:
    (1) valency: an integer scalar that indicates the actual number of valent
                 IDs in the current sample.
    (2) vectors: a 2D tensor of shape [max_valency, embedding_dim] that
                 represents the embedding lookup results to be combined. The
                 vectors are guaranteed to be in the same order as the embedding
                 IDs appear in the input sample.
    (3) weights: this argument is only present if `num_weights` of this class
                 is greater than 0. It is a 1D tensor of shape [num_weights]
                 that will be back-propagated to during the backward pass.
                 Currently only SGD optimizer is supported on these weights,
                 with the learning rate being the
                 `combiner_weights_learning_rate` argument of the constructor
                 of this class.

  The custom computation should return a 1D tensor of shape [embedding_dim] that
  represents the combined embedding vector. An example combiner computation is
  as follows (it simply sums over the valent embedding vectors and is
  semantically equivalent to the `sum` combiner):

  ```python
  @tf.function
  def sum_combiner(valency, vectors):
    max_valency = vectors.shape[0]
    valid_mask = tf.range(max_valency) < valency
    vectors_masked = tf.where(
        tf.expand_dims(valid_mask, axis=-1),
        vectors,
        tf.zeros_like(vectors),
    )
    return tf.reduce_sum(vectors_masked, axis=0)
  ```

  The custom computation is defined as a per-sample combiner function and it
  will be vectorized to the given batch size. This means certain constructs
  (such as control flow ops) may not be supported in the custom computation.

  NOTE: This combiner can only be used with the `TPUEmbeddingV2` class.

  This class can be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  combiner parameter to set a table specific combiner.

  ```python
  custom_combiner = tpu_embedding_v2_utils.CustomCombiner(
      sum_combiner,
      max_valency=16,
  )

  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      combiner=custom_combiner)
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...)
  ```
  In this example, the combiner of the first table will be the `sum_combiner`.
  The second table will use the default `mean` combiner.
  """

  CUSTOM_COMBINER_SLOT_NAME = "custom_combiner_variables"

  def __init__(
      self,
      combiner_computation: core.PolymorphicFunction,
      max_valency: int,
      num_weights: int = 0,
      initializer: Optional[init_ops_v2.Initializer] = None,
      combiner_weights_learning_rate: Union[float, Callable[[], float]] = 0.01,
  ) -> Any:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning an arbitrary value although a constructor returns
    # nothing; the generated name uses the hash of the computation, so it is not stable across
    # processes and two combiners over the same computation get the same name.
    # Inputs: combiner_computation (traced function taking valency, vectors and, when weights
    # exist, the weights); max_valency (integer maximum number of ids per sample);
    # num_weights (integer, default 0); initializer (required when num_weights is positive);
    # combiner_weights_learning_rate (float or callable, default 0.01). Raises a value error
    # if weights are requested without an initializer. Sets the combiner name and attributes.
    super().__init__()
    self.combiner = "custom_combiner_" + str(hash(combiner_computation))
    self.combiner_computation = combiner_computation
    self.max_valency = max_valency
    self.num_weights = num_weights
    self.combiner_weights_learning_rate = combiner_weights_learning_rate

    if num_weights > 0 and initializer is None:
      raise ValueError(
          "When `num_weights` is greater than 0, `initializer` must be set."
      )

    self._slot_names_attr = tuple([self.CUSTOM_COMBINER_SLOT_NAME])
    self._slot_initializers_attr = tuple(
        [initializer] if initializer is not None else ()
    )

  def _slot_names(self) -> List[Text]:
    # [FUNCTION DOC]
    # Returns a list with the single combiner variables slot name when the number of weights is
    # positive, otherwise an empty list.
    if self.num_weights > 0:
      return list(self._slot_names_attr)
    return []

  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    # [FUNCTION DOC]
    # Returns a list with the weight initializer when the number of weights is positive,
    # otherwise an empty list.
    if self.num_weights > 0:
      return list(self._slot_initializers_attr)
    return []

  def __str__(self) -> str:
    # [FUNCTION DOC]
    # Returns the generated combiner name string (prefix plus the computation hash).
    return self.combiner


@tf_export("tpu.experimental.embedding.QuantizationConfig")
class QuantizationConfig:
  # [CLASS DOC]
  # Settings for simulated quantization of embedding lookup results: number of buckets and the
  # lower and upper clipping bounds. Attached to a table configuration and written into the
  # optimization parameters message. Does not change the stored table type.
  """Settings for simulated quantization of the tpu embedding table.

  When simulated quantization is enabled, the results of the embedding lookup
  are clipped and quantized according to the settings here before the combiner
  is applied.

  For example, to quantize `input` the following is done:
  ```python
  if input < lower
    input = lower
  if input > upper
    input = upper
  quantum = (upper - lower) / (num_buckets - 1)
  input = math.floor((input - lower) / quantum + 0.5) * quantum + lower
  ```

  See tensorflow/core/protobuf/tpu/optimization_parameters.proto for more
  details.

  NOTE: This does not change the storage type of the embedding table, that will
  continue to be float32 as will the saved variable in the checkpoint. You will
  have to manually quantize the variable (typically with the same algorithm and
  settings as above) manually.
  """

  def __init__(self, num_buckets: int, lower: float, upper: float):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists a return value, but a constructor returns nothing; the
    # lower bound is not checked to be below the upper bound.
    # Inputs: num_buckets (integer, at least 2), lower and upper (floats, clipping range).
    # Raises a value error if num_buckets is below 2. Stores the three values.
    """Simulated quantizaiton configuration.

    Args:
      num_buckets: The number of quantization buckets, must be atleast 2.
      lower: The lower bound for the quantization range.
      upper: The upper bound for the quantization range.

    Returns:
      `QuantizationConfig`.

    Raises:
      ValueError: if `num_buckets` is less than 2.
    """
    if num_buckets < 2:
      raise ValueError(f"num_buckets is {num_buckets}, must be at least 2 for "
                       f"simulated quantization.")

    self.num_buckets = num_buckets
    self.lower = lower
    self.upper = upper

  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters):
    # [FUNCTION DOC]
    # HELPER: used by the table configuration when building a table descriptor.
    # Mutates the given parameters message: enables simulated quantization and sets the number
    # of buckets and the lower and upper clipping limits. Returns nothing.
    parameters.simulated_quantization.enabled = True
    parameters.simulated_quantization.num_buckets = self.num_buckets
    parameters.simulated_quantization.clipping_limits.lower.value = self.lower
    parameters.simulated_quantization.clipping_limits.upper.value = self.upper

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form listing number of buckets, lower and upper bounds.
    return ("QuantizationConfig(num_buckets={num_buckets!r}, lower={lower!r}, "
            "upper={upper!r})".format(
                num_buckets=self.num_buckets,
                lower=self.lower,
                upper=self.upper))


@tf_export("tpu.experimental.embedding.TableConfig")
class TableConfig:
  # [CLASS DOC]
  # Configuration of one embedding table: vocabulary size, embedding dimension, initializer,
  # optional table specific optimizer, combiner (string or custom combiner), name, optional
  # quantization config, optional precomputed layout and data type. Shared by several feature
  # configurations to share a table. Can write itself into a TPU embedding table descriptor.
  """Configuration data for one embedding table.

  This class holds the configuration data for a single embedding table. It is
  used as the `table` parameter of a
  `tf.tpu.experimental.embedding.FeatureConfig`. Multiple
  `tf.tpu.experimental.embedding.FeatureConfig` objects can use the same
  `tf.tpu.experimental.embedding.TableConfig` object. In this case a shared
  table will be created for those feature lookups.

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  """

  def __init__(
      self,
      vocabulary_size: int,
      dim: int,
      initializer: Optional[Callable[[Any], None]] = None,
      optimizer: Optional[_Optimizer] = None,
      combiner: Union[Text, CustomCombiner] = "mean",
      name: Optional[Text] = None,
      quantization_config: QuantizationConfig = None,
      # TODO(b/295372790): Change the type to SparseCoreTableLayout after it is
      # open sourced.
      layout: Optional[Any] = None,
      dtype: dtypes.DType = dtypes.float32,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists a return value (a constructor returns nothing) and
    # claims the name is required on SparseCore, but a missing name only logs a warning.
    # Inputs: vocabulary_size and dim (positive integers); initializer (optional callable,
    # defaults to a truncated normal with zero mean and standard deviation of one over the
    # square root of dim); optimizer (optional); combiner (one of the strings mean, sum, sqrtn,
    # or a custom combiner; default mean); name (optional); quantization_config (optional);
    # layout (optional); dtype (float32 or int32, default float32). Raises a value error for an
    # invalid size, dimension, non-callable initializer, unsupported combiner or dtype.
    # Side effect: logs a warning when no name is given.
    """Embedding table configuration.

    Args:
      vocabulary_size: Size of the table's vocabulary (number of rows).
      dim: The embedding dimension (width) of the table.
      initializer: A callable initializer taking one parameter, the shape of the
        variable that will be initialized. Will be called once per task, to
        initialize that task's shard of the embedding table. If not specified,
        defaults to `truncated_normal_initializer` with mean `0.0` and standard
        deviation `1/sqrt(dim)`.
      optimizer: An optional instance of an optimizer parameters class, instance
        of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. If set will override the global
        optimizer passed to `tf.tpu.experimental.embedding.TPUEmbedding`.
      combiner: A string or instance of a combiner class specifying how to
        reduce if there are multiple entries in a single row. Currently 'mean',
        'sqrtn', 'sum', and custom combiners are supported, with 'mean' the
        default. 'sqrtn' often achieves good accuracy, in particular with
        bag-of-words columns. For more information, see
        `tf.nn.embedding_lookup_sparse`.
      name: An optional string used to name the table. Must be defined if
        running on SparseCore.
      quantization_config: The simulated quantization config. An instance of
        `tf.tpu.experimental.embedding.QuantizationConfig`. See the class for
        more documentation.
      layout: If the table already has its layout computed, you can pass it in
        here. Otherwise, we will compute it for you. Most users should leave
        this as None.
      dtype: The data type of the embedding table. Currently only int32 or
        float32 is supported.

    Returns:
      `TableConfig`.

    Raises:
      ValueError: if `vocabulary_size` is not a positive integer.
      ValueError: if `dim` is not a positive integer.
      ValueError: if `initializer` is specified and is not callable.
      ValueError: if `combiner` is not supported.
    """
    if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
      raise ValueError(
          f"Argument `vocabulary_size` must be an int and must be >= 1. "
          f"Received: {vocabulary_size}")

    if not isinstance(dim, int) or dim < 1:
      raise ValueError(
          f"Argument `dim` (embedding dimension) "
          f"must be an int and must be >= 1. Received: {dim}")

    if (initializer is not None) and (not callable(initializer)):
      raise ValueError(
          f"Argument `initializer` must be a callable (or None). "
          f"Received: {initializer}")
    if initializer is None:
      initializer = init_ops_v2.TruncatedNormal(mean=0.0,
                                                stddev=1/math.sqrt(dim))

    accepted_str_combiners = ("mean", "sum", "sqrtn")
    if isinstance(combiner, str):
      if combiner not in accepted_str_combiners:
        raise ValueError(
            f"String argument `combiner` must be in {accepted_str_combiners}. "
            f"Received: {combiner}")

    elif not isinstance(combiner, CustomCombiner):
      raise ValueError(
          f"Argument `combiner` should either be a str or a CustomCombiner. "
          f"Received: {type(combiner)}"
      )

    if name is None:
      logging.warning(
          "Name of the table config must be specified for running on"
          " SparseCore. Different table configs must have unique names."
      )
    if dtype not in (dtypes.float32, dtypes.int32):
      raise ValueError(
          f"Argument `dtype` must be either float32 or int32. Received: {dtype}"
      )

    self.vocabulary_size = vocabulary_size
    self.dim = dim
    self.initializer = initializer
    self.optimizer = optimizer
    self.combiner = combiner
    self.name = name
    self.quantization_config = quantization_config
    self.layout = layout

    self.dtype = dtype

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form listing size, dimension, initializer, optimizer, combiner, name and
    # quantization config. The default truncated normal initializer is shown as nothing.
    # If using the default initializer, just print "None" for clarity.
    initializer = self.initializer

    if isinstance(initializer, init_ops_v2.TruncatedNormal):
      # PY2 type checking can't infer type of initializer even after if.
      initializer = typing.cast(init_ops_v2.TruncatedNormal, initializer)
      if (initializer.mean == 0.0
          and math.isclose(initializer.stddev, 1/math.sqrt(self.dim))):
        initializer = None

    return ("TableConfig(vocabulary_size={vocabulary_size!r}, dim={dim!r}, "
            "initializer={initializer!r}, optimizer={optimizer!r}, "
            "combiner={combiner!r}, name={name!r}, "
            "quantization_config={quantization!r})".format(
                vocabulary_size=self.vocabulary_size,
                dim=self.dim,
                initializer=initializer,
                optimizer=self.optimizer,
                combiner=self.combiner,
                name=self.name,
                quantization=self.quantization_config,
            ))

  def _set_table_descriptor(
      self,
      table_descriptor: tpu_embedding_configuration_pb2
      .TPUEmbeddingConfiguration.TableDescriptor,
      num_hosts: int,
      learning_rate_index: Dict[Callable[[], Any], int]):
    # [FUNCTION DOC]
    # HELPER: used by the embedding layer when building the TPU embedding configuration.
    # Mutates table_descriptor: sets the name, the vocabulary size padded up to num_hosts, the
    # dimension, and (if an optimizer is set) the learning rate (dynamic tag looked up in
    # learning_rate_index for callables, else constant), low dimensional packing, and the
    # optimizer's own parameters; then quantization parameters if configured. Returns nothing.
    """Set the table descriptor from the table data."""
    table_descriptor.name = self.name

    # For small tables, we pad to the number of hosts so that at least one
    # id will be assigned to each host.
    table_descriptor.vocabulary_size = max(self.vocabulary_size, num_hosts)
    table_descriptor.dimension = self.dim

    parameters = table_descriptor.optimization_parameters

    # We handle the learning rate separately here and don't allow the
    # optimization class to handle this, as it doesn't know about dynamic
    # rates.
    if self.optimizer:
      if callable(self.optimizer.learning_rate):
        parameters.learning_rate.dynamic.tag = (
            learning_rate_index[self.optimizer.learning_rate])
      else:
        parameters.learning_rate.constant = self.optimizer.learning_rate
      if self.optimizer.low_dimensional_packing_status:
        parameters.low_dimensional_packing_status = (
            optimization_parameters_pb2.LowDimensionalPackingStatus.Status.ENABLED
        )
      # Use optimizer to handle the rest of the parameters.
      self.optimizer._set_optimization_parameters(parameters)  # pylint: disable=protected-access

    if self.quantization_config:
      self.quantization_config._set_optimization_parameters(parameters)  # pylint: disable=protected-access


@tf_export("tpu.experimental.embedding.RowIdInitializer")
class RowIdInitializer:
  # [CLASS DOC]
  # Initializer that fills each table row with its row index plus an offset (every column of a
  # row holds the same value). Used to create tables whose contents identify the vocabulary id.
  """An initializer that initializes the table with vocabulary ids."""

  def __init__(self, offset: int = 0):
    # [FUNCTION DOC]
    # Stores the integer offset (default 0) added to each row index.
    self.offset = offset

  def __call__(
      self, shape: Union[Sequence[int], TensorShape], dtype: dtypes.DType
  ) -> core.Tensor:
    # [FUNCTION DOC]
    # Inputs: shape (table shape, first entry is the number of rows) and dtype. Returns a
    # tensor of that shape and dtype where row i holds the value offset plus i in all columns.
    return math_ops.range(
        start=self.offset, limit=self.offset + shape[0], delta=1, dtype=dtype
    )[:, None] * array_ops.ones(shape, dtype=dtype)


@tf_export("tpu.experimental.embedding.FeatureConfig")
class FeatureConfig:
  # [CLASS DOC]
  # Configuration of one embedding feature: the table configuration it is looked up in, an
  # optional maximum sequence length, a weights/indices validation flag, an optional output
  # shape (stored as a tensor shape) and an optional name. Used as the values of the feature
  # configuration structure passed to the TPU embedding layer.
  """Configuration data for one embedding feature.

  This class holds the configuration data for a single embedding feature. The
  main use is to assign features to `tf.tpu.experimental.embedding.TableConfig`s
  via the table parameter:

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  You can also specify the output shape for each feature. The output shape
  should be the expected activation shape excluding the table dimension. For
  dense and sparse tensor, the output shape should be the same as the input
  shape excluding the last dimension. For ragged tensor, the output shape can
  mismatch the input shape.

  NOTE: The `max_sequence_length` will be only used when the input tensor has
  rank 2 and the `output_shape` is not set in the feature config.

  When feeding features into `embedding.enqueue` they can be `tf.Tensor`s,
  `tf.SparseTensor`s or `tf.RaggedTensor`s. When the argument
  `max_sequence_length` is 0, the default, you should expect a output of
  `embedding.dequeue` for this feature of shape `(batch_size, dim)`. If
  `max_sequence_length` is greater than 0, the feature is embedded as a sequence
  and padded up to the given length. The shape of the output for this feature
  will be `(batch_size, max_sequence_length, dim)`.
  """

  def __init__(self,
               table: TableConfig,
               max_sequence_length: int = 0,
               validate_weights_and_indices: bool = True,
               output_shape: Optional[Union[List[int], TensorShape]] = None,
               name: Optional[Text] = None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists a return value, but a constructor returns nothing.
    # Inputs: table (a table configuration); max_sequence_length (non-negative integer, default
    # 0 meaning not a sequence); validate_weights_and_indices (boolean, default true);
    # output_shape (optional list or tensor shape); name (optional). Raises a value error if
    # table has the wrong type, max_sequence_length is not a non-negative integer, or the
    # validation flag is not a boolean. Stores the values.
    """Feature configuration.

    Args:
      table: An instance of `tf.tpu.experimental.embedding.TableConfig`,
        describing the table in which this feature should be looked up.
      max_sequence_length: If positive, the feature is a sequence feature with
        the corresponding maximum sequence length. If the sequence is longer
        than this, it will be truncated. If 0, the feature is not a sequence
        feature.
      validate_weights_and_indices: If true, uses safe_embedding_lookup during
        serving which ensures there are no empty rows and all weights and ids
        are positive at the expense of extra compute cost.
      output_shape: Optional argument to config the output shape of the feature
        activation. If provided, the feature feeding to the `embedding.enqueue`
        has to match the shape (for ragged tensor, the input shape and output
        shape can mismatch). If not provided, the shape can be either provided
        to the `embedding.build` or auto detected at the runtime.
      name: An optional string used to name the table. Must be defined if
        running on SparseCore.

    Returns:
      `FeatureConfig`.

    Raises:
      ValueError: if `table` is not an instance of
        `tf.tpu.experimental.embedding.TableConfig`.
      ValueError: if `max_sequence_length` not an integer or is negative.
    """
    if not isinstance(table, TableConfig):
      raise ValueError(f"Argument `table` has invalid type {type(table)}. "
                       "Expected `tf.tpu.experimental.embedding.TableConfig`.")

    if not isinstance(max_sequence_length, int) or max_sequence_length < 0:
      raise ValueError(
          f"Argument `max_sequence_length` must be an int and must be >= 0. "
          f"Received: {max_sequence_length}")

    self.table = table
    self.max_sequence_length = max_sequence_length
    self.name = name
    self.output_shape = TensorShape(output_shape)

    if not isinstance(
        validate_weights_and_indices, bool):
      raise ValueError(
          f"Argument `validate_weights_and_indices` must be a boolean. "
          f"Received: {validate_weights_and_indices}")

    self.validate_weights_and_indices = validate_weights_and_indices

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form listing table, maximum sequence length, validation flag, output shape
    # and name.
    return ("FeatureConfig(table={table!r}, "
            "max_sequence_length={max_sequence_length!r}, "
            "validate_weights_and_indices={validate_weights_and_indices!r}, "
            "output_shape={output_shape!r}, name={name!r})".format(
                table=self.table,
                max_sequence_length=self.max_sequence_length,
                validate_weights_and_indices=self.validate_weights_and_indices,
                output_shape=self.output_shape,
                name=self.name))


def log_tpu_embedding_configuration(
    config: tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration) -> None:
  # [FUNCTION DOC]
  # Logs a configuration message line by line at informational level, between a start and a
  # done message, so that long configurations are not truncated by the logger. Input: config,
  # the TPU embedding configuration protocol buffer. Returns nothing; side effect is logging.
  """Logs a TPUEmbeddingConfiguration proto across multiple statements.

  Args:
    config: TPUEmbeddingConfiguration proto to log.  Necessary because
      logging.info has a maximum length to each log statement, which
      particularly large configs can exceed.
  """
  logging.info("Beginning log of TPUEmbeddingConfiguration.")
  for line in str(config).splitlines():
    logging.info(line)
  logging.info("Done with log of TPUEmbeddingConfiguration.")


def _sort_device_spec_strings(device_strings: Iterable[str]) -> List[str]:
  # [FUNCTION DOC]
  # HELPER: gives a deterministic device order for host listing.
  # Input: device_strings, an iterable of device specification strings. Output: list of
  # normalized device strings sorted by replica, then task, then device index (numeric, not
  # alphabetical). Raises whatever the spec parser raises for malformed strings.
  sorted_specs = sorted(
      (device_spec.DeviceSpecV2.from_string(spec) for spec in device_strings),
      key=lambda s: (s.replica, s.task, s.device_index),
  )
  return [spec.to_string() for spec in sorted_specs]


def get_list_of_hosts(strategy: tpu_strategy.TPUStrategy) -> List[Text]:
  # [FUNCTION DOC]
  # Input: strategy, a TPU strategy. Output: list of unique host (CPU) device strings, in the
  # order of the sorted worker devices. Raises an assertion error if the number of hosts found
  # differs from the strategy's reported host count.
  """Returns a sorted list of CPU devices for the remote jobs.

  Args:
    strategy: A TPUStrategy object.

  Returns:
    A sorted list of device host strings.
  """

  list_of_hosts = []
  # Elsewehere we assume that the list of hosts is sorted.
  for tpu_device in _sort_device_spec_strings(strategy.extended.worker_devices):
    host = device_util.get_host_for_device(tpu_device)
    if host not in list_of_hosts:
      list_of_hosts.append(host)
  assert len(list_of_hosts) == strategy.extended.num_hosts
  return list_of_hosts
