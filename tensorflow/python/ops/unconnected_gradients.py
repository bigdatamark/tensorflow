# [MODULE DOC]
# Defines the UnconnectedGradients enumeration, exported publicly as the unconnected-gradients
# type. It is used by gradient computation APIs (gradient tapes and gradient functions) to choose
# what is returned for a source when the target does not depend on it: a null value or zeros.
# The only import-time side effect is registering the enum with the public API export machinery.
# Not a helper module; it is part of the public gradients API surface.
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

"""Utilities for calculating gradients."""
import enum

from tensorflow.python.util.tf_export import tf_export


@tf_export("UnconnectedGradients")
class UnconnectedGradients(enum.Enum):
  # [CLASS DOC]
  # Enumeration with two members: NONE (string value "none") means a null value is returned as
  # the gradient when no path connects target to source; ZERO (string value "zero") means a
  # zero tensor shaped like the source is returned instead. Plain enum with no custom behavior;
  # consumed by gradient APIs that accept an unconnected-gradients argument.
  """Controls how gradient computation behaves when y does not depend on x.

  The gradient of y with respect to x can be zero in two different ways: there
  could be no differentiable path in the graph connecting x to y (and so we can
  statically prove that the gradient is zero) or it could be that runtime values
  of tensors in a particular execution lead to a gradient of zero (say, if a
  relu unit happens to not be activated). To allow you to distinguish between
  these two cases you can choose what value gets returned for the gradient when
  there is no path in the graph from x to y:

  * `NONE`: Indicates that [None] will be returned if there is no path from x
    to y
  * `ZERO`: Indicates that a zero tensor will be returned in the shape of x.
  """
  NONE = "none"
  ZERO = "zero"
