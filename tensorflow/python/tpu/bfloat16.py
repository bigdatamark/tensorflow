# [MODULE DOC]
# Helper context for running models with bfloat16 variables on TPU. Provides a private custom
# variable getter that stores variables in 32-bit floating point but hands back 16-bit brain
# floating point values when that type is requested, and the public (version 1 only) scope
# function that installs this getter on a variable scope. Import-time side effect: registers the
# scope function in the public API under the tpu namespace for version 1.
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
# =============================================================================

"""Helper context for running models with bfloat16."""

from typing import Generator, Optional, Text

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


def _get_custom_getter():
  # [FUNCTION DOC]
  # HELPER: factory used by bfloat16_scope to build the custom variable getter.
  # KNOWN ISSUE: the docstring is stale (copied from a convolution network builder): it talks
  # about "this class's methods", a network builder, a use_tf_layers setting and a variable_type
  # attribute, none of which exist here. The function takes no arguments, is not a method, and
  # the returned getter always stores bfloat16 requests as float32 and casts back.
  # Takes no inputs and returns the inner getter function. No side effects.
  """Returns a custom getter that this class's methods must be called under.

  All methods of this class must be called under a variable scope that was
  passed this custom getter. Example:

  ```python
  network = ConvNetBuilder(...)
  with tf.compat.v1.variable_scope('cg',
                                   custom_getter=network.get_custom_getter()):
    network.conv(...)
    # Call more methods of network here
  ```

  Currently, this custom getter only does anything if self.use_tf_layers is
  True. In that case, it causes variables to be stored as dtype
  self.variable_type, then casted to the requested dtype, instead of directly
  storing the variable as the requested dtype.
  """

  def inner_custom_getter(getter, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: the custom getter returned by the enclosing factory.
    # KNOWN ISSUE: the docstring says it forces variables to the type self.variable_type, but
    # there is no such attribute; it only converts requested bfloat16 to float32 storage. It also
    # requires the dtype keyword argument to be present and raises a key error otherwise.
    # Inputs: getter (the underlying variable getter), plus positional and keyword arguments for
    # it, which must include the requested dtype. If the requested dtype is bfloat16, the request
    # is changed to float32 so precision is not lost, then the underlying getter is called; the
    # result is cast to bfloat16 in that case (the cast result is a tensor and not assignable).
    # Returns the variable, or the cast tensor. No other side effects.
    """Custom getter that forces variables to have type self.variable_type."""
    cast_to_bfloat16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == dtypes.bfloat16:
      # Only change the variable dtype if doing so does not decrease variable
      # precision.
      kwargs['dtype'] = dtypes.float32
      cast_to_bfloat16 = True
    var = getter(*args, **kwargs)
    # This if statement is needed to guard the cast, because batch norm
    # assigns directly to the return value of this custom getter. The cast
    # makes the return value not a variable so it cannot be assigned. Batch
    # norm variables are always in fp32 so this if statement is never
    # triggered for them.
    if cast_to_bfloat16:
      var = math_ops.cast(var, dtypes.bfloat16)
    return var

  return inner_custom_getter


@tf_export(v1=['tpu.bfloat16_scope'])
@tf_contextlib.contextmanager
def bfloat16_scope(
    name: Optional[Text] = None
) -> Generator[variable_scope.variable_scope, None, None]:
  # [FUNCTION DOC]
  # Context manager (exported for version 1 as the tpu bfloat16 scope) that opens a variable
  # scope using the bfloat16 custom getter, so variables created inside are stored as float32
  # but read as bfloat16 when that dtype is requested.
  # Input: name, optional scope name; nothing is treated as the empty name. Yields the opened
  # variable scope object. Side effect: enters and exits a variable scope in the current graph.
  """Scope class for bfloat16 variables so that the model uses custom getter.

  This enables variables to be read as bfloat16 type when using get_variable.

  Arguments:
    name: Name to use for scope.

  Yields:
    a variable scope.
  """
  if name is None:
    name = ''
  with variable_scope.variable_scope(
      name, custom_getter=_get_custom_getter()) as varscope:
    yield varscope
