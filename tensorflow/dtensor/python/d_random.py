# [MODULE DOC]
# DTensor-aware stateless random number generation and dropout. Offers stateless normal, uniform
# and truncated normal generators, a seed-splitting function, and a stateless dropout that all
# accept an optional layout so results can be created directly as sharded DTensors. The generators
# ignore the layout outside eager mode. Built on the DTensor api module and the stateless random
# ops. No import-time side effects.

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""DTensor helpers for random generators."""

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import shape_util

# ------------------------------------------------------------------------------
# stateless rngs
# ------------------------------------------------------------------------------


# TODO(b/171746536): switch all rng ops to official versions once supported.
def _old_tf_random_stateless_normal(
    shape,
    seed,
    mean=0.0,
    stddev=1.0,
    dtype=dtypes.float32,
    name=None,
    layout=None,
):
  # [FUNCTION DOC]
  # HELPER: implementation behind stateless_random_normal.
  # Inputs: shape (shape of the output); seed (integer pair seed, converted to a 32-bit integer
  # tensor); mean and stddev (scalars, default 0 and 1); dtype (default 32-bit float); name
  # (optional op name); layout (optional DTensor layout). Draws standard normal values through
  # the layout-aware call helper, scales by stddev and adds mean, and sets the static shape.
  # Returns the resulting tensor.
  """DTensor stateless normal implementation that takes an layout."""
  with ops.name_scope(
      name, "stateless_random_normal", [shape, seed, mean, stddev]
  ) as name:
    seed = ops.convert_to_tensor(seed, dtype=dtypes.int32, name="seed")
    shape = shape_util.shape_tensor(shape)
    mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    rnd = api.call_with_layout(
        gen_stateless_random_ops.stateless_random_normal,
        layout,
        shape,
        seed,
        dtype,
    )
    result = math_ops.add(rnd * stddev, mean, name=name)
    shape_util.maybe_set_static_shape(result, shape)
    return result


def _old_tf_random_stateless_uniform(
    shape,
    seed,
    minval=0,
    maxval=None,
    dtype=dtypes.float32,
    name=None,
    layout=None,
):
  # [FUNCTION DOC]
  # HELPER: implementation behind stateless_random_uniform.
  # Inputs: shape; seed (integer pair); minval (default 0); maxval (default nothing, meaning 1 for
  # floats and the full integer range for integers); dtype (default 32-bit float); name (optional);
  # layout (optional DTensor layout). Raises a value error if dtype is not one of float16, bfloat16,
  # float32, float64, int32, int64, uint32, uint64; if for integer types only one of minval/maxval
  # is given; or if minval is given for an unsigned type. Integer types with no bounds use the
  # full-range generator; integer types with bounds use the ranged integer generator; floats scale a
  # unit uniform value by the range and add minval. Sets the static shape and returns the tensor.
  # KNOWN ISSUE: for integer types the converted range tensor is computed but unused, and minval and
  # maxval are passed as given to the op.
  """DTensor stateless uniform implementation that takes an layout."""
  dtype = dtypes.as_dtype(dtype)
  accepted_dtypes = (
      dtypes.float16,
      dtypes.bfloat16,
      dtypes.float32,
      dtypes.float64,
      dtypes.int32,
      dtypes.int64,
      dtypes.uint32,
      dtypes.uint64,
  )
  if dtype not in accepted_dtypes:
    raise ValueError(
        f"Argument `dtype` got invalid value {dtype}. Accepted dtypes are "
        f"{accepted_dtypes}."
    )
  if dtype.is_integer:
    if (minval is None) != (maxval is None):
      raise ValueError(
          f"For integer `dtype` argument {dtype}, argument `minval` and "
          f"`maxval` must be both None or not None. Got `minval`={minval} and "
          f"`maxval`={maxval}."
      )
    if minval is not None and dtype in (dtypes.uint32, dtypes.uint64):
      raise ValueError(
          f"Argument `dtype` got invalid value {dtype} when argument `minval` "
          "is not None. Please don't use unsigned integers in this case."
      )

  shape = shape_util.shape_tensor(shape)
  with ops.name_scope(
      name, "stateless_random_uniform", [shape, seed, minval, maxval]
  ) as name:
    seed = ops.convert_to_tensor(seed, dtype_hint=dtypes.int32, name="seed")

    if dtype.is_integer and minval is None and maxval is None:
      result = api.call_with_layout(
          gen_stateless_random_ops.stateless_random_uniform_full_int,
          layout,
          shape,
          seed=seed,
          dtype=dtype,
          name=name,
      )
    else:
      if not dtype.is_integer and maxval is None:
        maxval = 1
      val_range = ops.convert_to_tensor(
          maxval - minval, dtype=dtype, name="range"
      )
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      if dtype.is_integer:
        result = api.call_with_layout(
            gen_stateless_random_ops.stateless_random_uniform_int,
            layout,
            shape,
            seed=seed,
            minval=minval,
            maxval=maxval,
        )
      else:
        rnd = api.call_with_layout(
            gen_stateless_random_ops.stateless_random_uniform,
            layout,
            shape,
            seed=seed,
            dtype=dtype,
        )
        result = math_ops.add(rnd * val_range, minval, name=name)
    shape_util.maybe_set_static_shape(result, shape)
    return result


def _old_tf_stateless_truncated_normal(
    shape,
    seed,
    mean=0.0,
    stddev=1.0,
    dtype=dtypes.float32,
    name=None,
    layout=None,
):
  # [FUNCTION DOC]
  # HELPER: implementation behind stateless_truncated_normal.
  # Same inputs as the normal generator helper (shape, seed, mean default 0, stddev default 1, dtype
  # default 32-bit float, name, layout). Draws truncated normal values through the layout-aware call
  # helper, scales by stddev, adds mean, sets the static shape and returns the tensor.
  """DTensor stateless truncated normal implementation that takes an layout."""
  with ops.name_scope(
      name, "stateless_truncated_normal", [shape, seed, mean, stddev]
  ) as name:
    seed = ops.convert_to_tensor(seed, dtype=dtypes.int32, name="seed")
    shape = shape_util.shape_tensor(shape)
    mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    rnd = api.call_with_layout(
        gen_stateless_random_ops.stateless_truncated_normal,
        layout,
        shape,
        seed,
        dtype,
    )
    result = math_ops.add(rnd * stddev, mean, name=name)
    shape_util.maybe_set_static_shape(result, shape)
    return result


def stateless_random_normal(
    shape,
    seed,
    mean=0.0,
    stddev=1.0,
    dtype=dtypes.float32,
    name=None,
    layout=None,
):
  # [FUNCTION DOC]
  # Stateless normal random generator. Inputs: shape, seed (pair), mean (default 0), stddev
  # (default 1), dtype (default 32-bit float), name (optional), layout (optional DTensor layout).
  # The layout is dropped (treated as nothing) when not executing eagerly. Returns a normally
  # distributed tensor, sharded per the layout if given. Delegates to the private helper.
  """DTensor stateless RNG."""
  if not context.executing_eagerly():
    layout = None

  return _old_tf_random_stateless_normal(
      shape,
      seed=seed,
      mean=mean,
      stddev=stddev,
      dtype=dtype,
      name=name,
      layout=layout,
  )


def stateless_random_uniform(
    shape,
    seed,
    minval=0,
    maxval=None,
    dtype=dtypes.float32,
    name=None,
    layout=None,
):
  # [FUNCTION DOC]
  # Stateless uniform random generator. Inputs: shape, seed (pair), minval (default 0), maxval
  # (default nothing), dtype (default 32-bit float), name (optional), layout (optional DTensor
  # layout). The layout is dropped when not executing eagerly. Returns a uniformly distributed
  # tensor; see the private helper for dtype validation errors it raises.
  """DTensor stateless random uniform."""
  if not context.executing_eagerly():
    layout = None

  return _old_tf_random_stateless_uniform(
      shape,
      seed=seed,
      minval=minval,
      maxval=maxval,
      dtype=dtype,
      name=name,
      layout=layout,
  )


def stateless_truncated_normal(
    shape,
    seed,
    mean=0.0,
    stddev=1.0,
    dtype=dtypes.float32,
    name=None,
    layout=None,
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring is the generic "stateless RNG" text; this specifically generates
  # truncated normal values.
  # Stateless truncated normal generator. Inputs: shape, seed (pair), mean (default 0), stddev
  # (default 1), dtype (default 32-bit float), name (optional), layout (optional DTensor layout).
  # The layout is dropped when not executing eagerly. Returns a truncated normal tensor.
  """DTensor stateless RNG."""
  if not context.executing_eagerly():
    layout = None

  return _old_tf_stateless_truncated_normal(
      shape,
      seed=seed,
      mean=mean,
      stddev=stddev,
      dtype=dtype,
      name=name,
      layout=layout,
  )


def stateless_split(seed, num=2, mesh=None):
  # [FUNCTION DOC]
  # Splits a seed into several new seeds. Inputs: seed (integer pair, converted to a tensor); num
  # (number of new seeds, default 2); mesh (optional mesh; when given, the result is fully
  # replicated on it). Draws a num-by-2 matrix of full-range random integers of the seed's dtype
  # via the stateless uniform generator. Returns that tensor.
  seed = ops.convert_to_tensor(seed)
  layout = None
  if mesh:
    layout = layout_lib.Layout.replicated(mesh, rank=2)
  return stateless_random_uniform(
      shape=[num, 2],
      seed=seed,
      dtype=seed.dtype,
      minval=None,
      maxval=None,
      layout=layout,
  )


# ------------------------------------------------------------------------------
# stateless dropout.
# ------------------------------------------------------------------------------


def _get_noise_shape(x, noise_shape):
  # [FUNCTION DOC]
  # HELPER: computes the shape of the dropout noise tensor (copied from the core nn ops).
  # Inputs: x (tensor being dropped out); noise_shape (nothing, or a shape-like value). With nothing
  # returns the dynamic shape of x. If noise_shape cannot be converted to a shape, returns it
  # unchanged for the op to handle. If ranks match, unknown noise dimensions are filled in with the
  # static dimensions of x, and a shape object is returned; otherwise returns noise_shape as given.
  """Noisve shape util copied from tf nn_ops."""
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape


# TODO(b/171213877, b/169909066): Fix layout prop in function case for the rng
# Op used. The layout prop should be able to propagate the layout from input
# tensor `x` to the tf.mul and then back propagate the layout to the
# `random_tensor`.
def dropout(x, rate, noise_shape=None, seed=None, name=None):
  # [FUNCTION DOC]
  # Stateless DTensor replacement for dropout. Inputs: x (tensor, typically a DTensor); rate
  # (fraction to drop, must be a float); noise_shape (optional shape of the random mask, default the
  # shape of x); seed (required integer; combined with zero into a pair); name (optional).
  # Raises a value error if rate is not a float or seed is nothing. Scales x by one over the keep
  # probability, draws a uniform mask using the layout of x (eager only; none in graph mode), keeps
  # elements whose random value is at least rate, multiplies by the mask cast to the dtype of x, and
  # restores the static shape in graph mode. Returns the resulting tensor.
  """DTensor replacement for dropout."""
  if not isinstance(rate, float):
    raise ValueError("rate should be float for dropout.")
  if seed is None:
    raise ValueError("seed must be specified for DTensor dropout. Got: None")

  with ops.name_scope(name, "dropout", [x]):
    x_dtype = x.dtype
    keep_prob = 1 - rate
    scale = 1 / keep_prob
    scale = ops.convert_to_tensor(scale, dtype=x_dtype)
    ret = gen_math_ops.mul(x, scale)

    noise_shape = _get_noise_shape(x, noise_shape)
    # stateless_random_uniform requires a shape [2] seed.
    seed = [seed, 0]

    if context.executing_eagerly():
      layout = api.fetch_layout(x)
    else:
      layout = None
    random_tensor = _old_tf_random_stateless_uniform(
        noise_shape, seed=seed, minval=0, maxval=1, dtype=x_dtype, layout=layout
    )
    keep_mask = random_tensor >= rate
    ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
    if not context.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret


# TODO(b/195413777): error out for stateful dropout.
