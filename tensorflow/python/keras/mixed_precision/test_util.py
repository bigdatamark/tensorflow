# [MODULE DOC]
# HELPER: Shared testing utilities for the Keras mixed precision tests (not production code).
# Provides factories for identity functions with gradient assertions or injected NaN gradients,
# and small test layers (a type-asserting base layer, a layer multiplying its input by a scalar
# variable, a variant without automatic variable casting) plus two trivial regularizers.
# No import-time side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains testing utilities related to mixed precision."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


def create_identity_with_grad_check_fn(expected_gradient, expected_dtype=None):
  # [FUNCTION DOC]
  # HELPER: Test utility. Inputs: expected_gradient (value the incoming gradient must equal)
  # and expected_dtype (optional dtype the gradient must have). Returns a plain (undecorated,
  # so Keras lambda layers can serialize it) identity function whose custom gradient asserts
  # the dtype (when given) and, through a graph assertion ordered after the input, that the
  # gradient equals the expected value, then passes the gradient through. Assertion failures
  # occur later, during gradient computation, not when this factory runs.
  """Returns a function that asserts it's gradient has a certain value.

  This serves as a hook to assert intermediate gradients have a certain value.
  This returns an identity function. The identity's gradient function is also
  the identity function, except it asserts that the gradient equals
  `expected_gradient` and has dtype `expected_dtype`.

  Args:
    expected_gradient: The gradient function asserts that the gradient is this
      value.
    expected_dtype: The gradient function asserts the gradient has this dtype.

  Returns:
    An identity function whose gradient function asserts the gradient has a
    certain value.
  """
  @custom_gradient.custom_gradient
  def _identity_with_grad_check(x):
    # [FUNCTION DOC]
    # HELPER: Custom-gradient identity. Input: tensor x. Returns x passed through an identity
    # operation together with the gradient function defined below.
    """Function that asserts it's gradient has a certain value."""
    x = array_ops.identity(x)
    def grad(dx):
      # [FUNCTION DOC]
      # HELPER: Gradient function. Input: upstream gradient dx. Fails with an assertion error
      # if an expected dtype was given and dx differs. Builds a graph assertion (after the
      # input is available, so end-of-data errors skip it) that dx equals the expected value
      # converted to dx's dtype. Returns dx through an identity ordered after the assertion.
      """Gradient function that asserts the gradient has a certain value."""
      if expected_dtype:
        assert dx.dtype == expected_dtype, (
            'dx.dtype should be %s but is: %s' % (expected_dtype, dx.dtype))
      expected_tensor = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          expected_gradient, dtype=dx.dtype, name='expected_gradient'
      )
      # Control dependency is to ensure input is available. It's possible the
      # dataset will throw a StopIteration to indicate there is no more data, in
      # which case we don't want to run the assertion.
      with ops.control_dependencies([x]):
        assert_op = check_ops.assert_equal(dx, expected_tensor)
      with ops.control_dependencies([assert_op]):
        dx = array_ops.identity(dx)
      return dx
    return x, grad
  # Keras sometimes has trouble serializing Lambda layers with a decorated
  # function. So we define and return a non-decorated function.
  def identity_with_grad_check(x):
    # [FUNCTION DOC]
    # HELPER: Undecorated wrapper returned to callers. Input x; returns the result of the
    # custom-gradient identity applied to x.
    return _identity_with_grad_check(x)
  return identity_with_grad_check


def create_identity_with_nan_gradients_fn(have_nan_gradients):
  # [FUNCTION DOC]
  # HELPER: Test utility. Input: have_nan_gradients, a scalar boolean tensor. Returns a plain
  # identity function whose custom gradient yields the incoming gradient multiplied by
  # not-a-number when the tensor is true at run time, and the unchanged gradient otherwise.
  # Used to simulate overflowing gradients for loss scaling tests.
  """Returns a function that optionally has NaN gradients.

  This serves as a hook to introduce NaN gradients to a model. This returns an
  identity function. The identity's gradient function will check if the boolean
  tensor `have_nan_gradients` is True. If so, the gradient will be NaN.
  Otherwise, the gradient will also be the identity.

  Args:
    have_nan_gradients: A scalar boolean tensor. If True, gradients will be NaN.
      Otherwise, the gradient function is the identity function.

  Returns:
    An identity function whose gradient function will return NaNs, if
    `have_nan_gradients` is True.
  """
  @custom_gradient.custom_gradient
  def _identity_with_nan_gradients(x):
    # [FUNCTION DOC]
    # HELPER: Custom-gradient identity. Input: tensor x. Returns an identity copy of x and the
    # gradient function defined below.
    """Function whose gradient is NaN iff `have_nan_gradients` is True."""
    x = array_ops.identity(x)
    def grad(dx):
      # [FUNCTION DOC]
      # HELPER: Gradient function. Input: upstream gradient dx. Returns a conditional that
      # evaluates to dx times not-a-number if the captured boolean is true, else dx unchanged.
      return cond.cond(
          have_nan_gradients,
          lambda: dx * float('NaN'),
          lambda: dx
      )
    return x, grad
  # Keras sometimes has trouble serializing Lambda layers with a decorated
  # function. So we define and return a non-decorated function.
  def identity_with_nan_gradients(x):
    # [FUNCTION DOC]
    # HELPER: Undecorated wrapper returned to callers. Input x; returns the result of the
    # custom-gradient identity applied to x.
    return _identity_with_nan_gradients(x)
  return identity_with_nan_gradients


class AssertTypeLayer(base_layer.Layer):
  # [CLASS DOC]
  # HELPER: Test base layer. Stores an optional dtype name (assert_type) and offers a method
  # that subclasses call inside their call method to assert that all input tensors have that
  # dtype. With no assert type configured the check is disabled.
  """A layer which asserts it's inputs are a certain type."""

  def __init__(self, assert_type=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: assert_type (optional dtype or dtype name; stored as the dtype name
    # string, or nothing if not given) and keyword arguments passed to the base layer
    # constructor. Returns nothing.
    self._assert_type = (dtypes.as_dtype(assert_type).name if assert_type
                         else None)
    super(AssertTypeLayer, self).__init__(**kwargs)

  def assert_input_types(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Input: inputs, a tensor or nested structure of tensors. If an assert type is
    # configured, flattens the structure and raises an assertion error for any tensor whose
    # base dtype differs from it, with a message naming both types. Otherwise does nothing.
    """Asserts `inputs` are of the correct type. Should be called in call()."""
    if self._assert_type:
      inputs_flattened = nest.flatten(inputs)
      for inp in inputs_flattened:
        assert inp.dtype.base_dtype == self._assert_type, (
            'Input tensor has type %s which does not match assert type %s' %
            (inp.dtype.name, self._assert_type))


class MultiplyLayer(AssertTypeLayer):
  # [CLASS DOC]
  # HELPER: Test layer that multiplies its input by a single scalar weight (initialized to
  # one), optionally with weight and activity regularizers, choosing either the multiply
  # operator or the multiply function, and asserting input dtypes. Used to check dtype
  # policies, variable autocasting and serialization round trips through its configuration.
  """A layer which multiplies its input by a scalar variable."""

  def __init__(self,
               regularizer=None,
               activity_regularizer=None,
               use_operator=False,
               var_name='v',
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: regularizer and activity_regularizer (regularizer objects, nothing,
    # or serialized mappings which are deserialized using this module's global names as custom
    # objects), use_operator (bool, default false: use the multiply operator instead of the
    # multiply function), var_name (name of the weight, default v) and keyword arguments for
    # the parent layer (such as the assert type or dtype). Stores them, then initializes the
    # parent with the activity regularizer. Returns nothing.
    """Initializes the MultiplyLayer.

    Args:
      regularizer: The weight regularizer on the scalar variable.
      activity_regularizer: The activity regularizer.
      use_operator: If True, add using the * operator. If False, add using
        tf.multiply.
      var_name: The name of the variable. It can be useful to pass a name other
        than 'v', to test having the attribute name (self.v) being different
        from the variable name.
      **kwargs: Passed to AssertTypeLayer constructor.
    """
    self._regularizer = regularizer
    if isinstance(regularizer, dict):
      self._regularizer = regularizers.deserialize(regularizer,
                                                   custom_objects=globals())
    self._activity_regularizer = activity_regularizer
    if isinstance(activity_regularizer, dict):
      self._activity_regularizer = regularizers.deserialize(
          activity_regularizer, custom_objects=globals())

    self._use_operator = use_operator
    self._var_name = var_name
    super(MultiplyLayer, self).__init__(
        activity_regularizer=self._activity_regularizer, **kwargs)

  def build(self, _):
    # [FUNCTION DOC]
    # Input shape is ignored. Creates the scalar weight (initialized to ones, using the
    # configured name and weight regularizer), stores it as the attribute v and marks the
    # layer built. Returns nothing.
    self.v = self.add_weight(
        self._var_name, (), initializer='ones', regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a tensor. Asserts its dtype matches the assert type, then returns
    # the input multiplied by the scalar weight.
    self.assert_input_types(inputs)
    return self._multiply(inputs, self.v)

  def _multiply(self, x, y):
    # [FUNCTION DOC]
    # HELPER: Inputs: two tensors x and y. Returns their product using the multiply operator if
    # the layer was configured with the operator option, otherwise using the multiply function.
    if self._use_operator:
      return x * y
    else:
      return math_ops.multiply(x, y)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent layer configuration extended with serialized regularizer and activity
    # regularizer, the operator flag, the variable name and the assert type, so the layer can
    # be re-created from its configuration.
    config = super(MultiplyLayer, self).get_config()
    config['regularizer'] = regularizers.serialize(self._regularizer)
    config['activity_regularizer'] = regularizers.serialize(
        self._activity_regularizer)
    config['use_operator'] = self._use_operator
    config['var_name'] = self._var_name
    config['assert_type'] = self._assert_type
    return config


class MultiplyLayerWithoutAutoCast(MultiplyLayer):
  # [CLASS DOC]
  # HELPER: Test variant of MultiplyLayer whose weight is created with automatic casting
  # disabled (always float32 or float64) and is cast manually to the input dtype in call.
  """Same as MultiplyLayer, but does not use AutoCastVariables."""

  def build(self, _):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The weight is always named v; the var_name constructor argument stored by
    # the parent is ignored here.
    # Input shape is ignored. Uses the layer dtype, replacing 16-bit float dtypes by float32,
    # creates the scalar weight initialized to ones with autocasting disabled and the weight
    # regularizer, stores it as v and marks the layer built.
    dtype = self.dtype
    if dtype in ('float16', 'bfloat16'):
      dtype = 'float32'
    self.v = self.add_weight(
        'v', (),
        initializer='ones',
        dtype=dtype,
        experimental_autocast=False,
        regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    # [FUNCTION DOC]
    # Forward pass. Input: a tensor. Asserts its dtype matches the assert type and that the
    # weight is float32 or float64, then returns the input multiplied by the weight cast to the
    # input dtype.
    self.assert_input_types(inputs)
    assert self.v.dtype in (dtypes.float32, dtypes.float64)
    return self._multiply(inputs, math_ops.cast(self.v, inputs.dtype))


class IdentityRegularizer(regularizers.Regularizer):
  # [CLASS DOC]
  # HELPER: Test regularizer that asserts its input is float32 and returns it unchanged as the
  # penalty. Has an empty configuration.

  def __call__(self, x):
    # [FUNCTION DOC]
    # Input: tensor x. Raises an assertion error if it is not float32; otherwise returns an
    # identity copy of it.
    assert x.dtype == dtypes.float32
    return array_ops.identity(x)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns an empty configuration mapping.
    return {}


class ReduceSumRegularizer(regularizers.Regularizer):
  # [CLASS DOC]
  # HELPER: Test regularizer whose penalty is the sum of all elements of its input. Has an
  # empty configuration.

  def __call__(self, x):
    # [FUNCTION DOC]
    # Input: tensor x. Returns the sum of all its elements as a scalar.
    return math_ops.reduce_sum(x)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns an empty configuration mapping.
    return {}
