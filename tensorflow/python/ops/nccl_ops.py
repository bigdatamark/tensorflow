# [MODULE DOC]
# Python wrappers for NVIDIA NCCL collective operations on GPUs: all-reduce (sum, product,
# minimum, maximum), reduce (sum only) and broadcast. Public functions take a list of tensors that
# must each already be assigned to a GPU device and return tensors placed on the same devices.
# The module also registers gradient functions for the NcclAllReduce, NcclReduce and
# NcclBroadcast ops (only sum reduction is differentiable). Private helpers validate device
# placement and generate unique shared names that tie together the per-device kernels of one
# collective. Import-time side effects: registers the three gradients and creates a module lock
# and a counter used for shared names. In eager mode all-reduce is wrapped in a traced function
# because NCCL kernels block unless they run concurrently.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for GPU collective operations implemented using NVIDIA nccl."""
import threading

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops


_module_lock = threading.Lock()
_shared_name_counter = 0


def all_sum(tensors):
  # [FUNCTION DOC]
  # Public entry point for NCCL all-reduce with sum. Takes a non-empty list of tensors, each
  # placed on a distinct GPU, and returns a list of the same length where every element holds the
  # elementwise sum of all inputs, located on the device of the corresponding input. Raises a
  # value error if the list is empty or a tensor has no device assignment. All outputs must be
  # evaluated or the collective hangs. Delegates to the shared all-reduce helper.
  """Returns a list of tensors with the all-reduce sum across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to sum; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the sum of the input tensors, where tensor i has
    the same device as `tensors[i]`.
  """
  return _apply_all_reduce('sum', tensors)


@ops.RegisterGradient('NcclAllReduce')
def _all_sum_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the NcclAllReduce op.
  # Computes the gradient of an all-reduce sum by running another all-reduce sum over the
  # incoming gradient on the same device, with the shared name of the forward op plus a gradient
  # suffix and the same device count. Inputs: the forward operation and the gradient with respect
  # to its output. Returns the gradient tensor for the op input. Raises a lookup error if the
  # forward reduction is not sum; raises a value error if the gradient lacks a device or sits on
  # a device different from the op.
  """The gradients for `all_sum`.

  Args:
    op: The `all_sum` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `all_sum` op.

  Returns:
    The gradient with respect to the output of `all_sum`.

  Raises:
    LookupError: If `reduction` is not `sum`.
  """
  if op.get_attr('reduction') != b'sum':
    raise LookupError('No gradient defined for NcclAllReduce except for '
                      'reduction="sum".')

  _check_device(grad, expected=op.device)
  num_devices = op.get_attr('num_devices')
  shared_name = op.get_attr('shared_name') + b'_grad'

  with ops.device(op.device):
    return gen_nccl_ops.nccl_all_reduce(
        input=grad,
        reduction='sum',
        num_devices=num_devices,
        shared_name=shared_name)


def all_prod(tensors):
  # [FUNCTION DOC]
  # Public NCCL all-reduce with product. Takes a non-empty list of GPU-placed tensors and returns
  # a list in which each element is the elementwise product of all inputs, on the device of the
  # matching input. Raises a value error for an empty list or a tensor without a device. Outputs
  # must all be evaluated or the collective hangs. No gradient is registered for this reduction.
  """Returns a list of tensors with the all-reduce product across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to multiply; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the product of the input tensors, where tensor i
    has the same device as `tensors[i]`.
  """
  return _apply_all_reduce('prod', tensors)


def all_min(tensors):
  # [FUNCTION DOC]
  # Public NCCL all-reduce with elementwise minimum. Takes a non-empty list of GPU-placed tensors
  # and returns a list in which each element is the minimum across all inputs, placed on the
  # device of the matching input. Raises a value error for an empty list or a missing device.
  # Outputs must all be evaluated or the collective hangs. No gradient is registered.
  """Returns a list of tensors with the all-reduce min across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to reduce; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the minimum of the input tensors, where tensor i
    has the same device as `tensors[i]`.
  """
  return _apply_all_reduce('min', tensors)


def all_max(tensors):
  # [FUNCTION DOC]
  # Public NCCL all-reduce with elementwise maximum. Takes a non-empty list of GPU-placed tensors
  # and returns a list in which each element is the maximum across all inputs, placed on the
  # device of the matching input. Raises a value error for an empty list or a missing device.
  # Outputs must all be evaluated or the collective hangs. No gradient is registered.
  """Returns a list of tensors with the all-reduce max across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to reduce; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the maximum of the input tensors, where tensor i
    has the same device as `tensors[i]`.
  """
  return _apply_all_reduce('max', tensors)


def reduce_sum(tensors):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a lookup error is raised when the context is not on a GPU,
  # but the code raises value errors instead (empty input, missing device assignment, or no
  # input tensor on the device where the result lands); no lookup error is raised here.
  # Public NCCL reduce with sum. Takes a non-empty list of GPU-placed tensors and returns a single
  # tensor holding their elementwise sum, located on the current device. Delegates to the reduce
  # helper.
  """Returns a tensor with the reduce sum across `tensors`.

  The computation is done with a reduce operation, so only one tensor is
  returned.

  Args:
    tensors: The input tensors across which to sum; must be assigned
      to GPU devices.

  Returns:
    A tensor containing the sum of the input tensors.

  Raises:
    LookupError: If context is not currently using a GPU device.
  """
  return _apply_reduce('sum', tensors)


@ops.RegisterGradient('NcclReduce')
def _reduce_sum_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the NcclReduce op.
  # KNOWN ISSUE: the error message for a non-sum reduction mentions NcclAllReduce although this
  # is the gradient of NcclReduce (copy-paste wording).
  # Gradient of a reduce sum: broadcasts the incoming gradient with the NCCL broadcast op on the
  # op device and returns that same tensor once per input of the forward op. Raises a lookup
  # error if the forward reduction is not sum; raises a value error if the gradient has no device
  # or is on a different device than the op.
  """The gradients for input `Operation` of `reduce_sum`.

  Args:
    op: The `sum send` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `reduce_sum` op.

  Returns:
    The gradient with respect to the input of `reduce_sum` op.

  Raises:
    LookupError: If the reduction attribute of op is not `sum`.
  """
  if op.get_attr('reduction') != b'sum':
    raise LookupError('No gradient defined for NcclAllReduce except for '
                      'reduction="sum".')
  _check_device(grad, expected=op.device)

  with ops.device(op.device):
    result = gen_nccl_ops.nccl_broadcast(input=grad, shape=grad.shape)

  return [result] * len(op.inputs)


def broadcast(tensor):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring refers to a value named source tensor, but the parameter is named
  # tensor.
  # Public NCCL broadcast sender. Validates that the single input tensor has a device, then
  # builds an NCCL broadcast op on that device and returns its output tensor, which other GPU
  # devices may consume. Raises a value error if the tensor has no device assignment.
  """Returns a tensor that can be efficiently transferred to other devices.

  Args:
    tensor: The tensor to send; must be assigned to a GPU device.

  Returns:
    A tensor with the value of `src_tensor`, which can be used as input to
    ops on other GPU devices.
  """
  _check_device(tensor)

  with ops.device(tensor.device):
    return gen_nccl_ops.nccl_broadcast(input=tensor, shape=tensor.shape)


@ops.RegisterGradient('NcclBroadcast')
def _broadcast_grad(op, accumulated_grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the NcclBroadcast op.
  # Gradient of broadcast: assumes the accumulated gradient was produced by an addition of the
  # per-receiver gradients, takes that producing op's inputs, checks each has a device, and sums
  # them with an NCCL reduce on the broadcast op device. Returns that reduced tensor. Raises a
  # value error if any input lacks a device assignment.
  """The gradients for input `Operation` of `broadcast`.

  Args:
    op: The `broadcast send` `Operation` that we are differentiating.
    accumulated_grad: Accumulated gradients with respect to the output of the
      `broadcast` op.

  Returns:
    Gradients with respect to the input of `broadcast`.
  """
  # Grab inputs of accumulated_grad and replace accumulation with reduce_sum.
  grads = [t for t in accumulated_grad.op.inputs]
  for t in grads:
    _check_device(t)

  with ops.device(op.device):
    return gen_nccl_ops.nccl_reduce(input=grads, reduction='sum')


def _apply_all_reduce(reduction, tensors):
  # [FUNCTION DOC]
  # HELPER: shared implementation for the public all-reduce functions.
  # Inputs: reduction name (sum, prod, min or max) and a non-empty list of tensors. Raises a value
  # error if the list is empty. Allocates a fresh shared name so the per-device ops form one
  # collective, then builds one all-reduce op per tensor on that tensor device. When executing
  # eagerly it wraps the construction in a traced function and calls it so the ops run
  # concurrently; otherwise it builds them directly in the current graph. Returns the list of
  # output tensors.
  """Helper function for all_* functions."""
  if not tensors:
    raise ValueError('Must pass >0 tensors to all reduce operations')

  shared_name = _get_shared_name()

  def _all_reduce():
    # [FUNCTION DOC]
    # HELPER: inner builder used by the all-reduce helper, directly or inside a traced function.
    # Takes no arguments; uses the enclosing tensors, reduction and shared name. For each tensor
    # it checks the device (value error if unassigned) and creates an NCCL all-reduce op on that
    # device with the device count equal to the number of tensors. Returns the list of outputs.
    """Call nccl allreduce."""
    res = []
    for t in tensors:
      _check_device(t)
      with ops.device(t.device):
        res.append(
            gen_nccl_ops.nccl_all_reduce(
                input=t,
                reduction=reduction,
                num_devices=len(tensors),
                shared_name=shared_name))
    return res

  if context.executing_eagerly():
    # Nccl ops will block unless they are executed concurrently such as in a
    # graph or a defun.
    return def_function.function(_all_reduce)()
  else:
    return _all_reduce()


def _apply_reduce(reduction, tensors):
  # [FUNCTION DOC]
  # HELPER: shared implementation for the public reduce functions.
  # Inputs: reduction name and a non-empty list of tensors. Raises a value error if the list is
  # empty, if any tensor lacks a device, or if no input tensor is on the same device as the
  # reduce result (the current device). Creates one NCCL reduce op over all inputs and returns
  # its single output tensor.
  """Helper function for reduce_* functions."""
  if not tensors:
    raise ValueError('Must pass >0 tensors to reduce operations')

  for t in tensors:
    _check_device(t)
  result = gen_nccl_ops.nccl_reduce(input=tensors, reduction=reduction)
  try:
    next(t for t in tensors if t.device == result.device)
  except StopIteration:
    raise ValueError('One input tensor must be assigned to current device')
  return result


def _get_shared_name():
  # [FUNCTION DOC]
  # HELPER: produces unique collective identifiers for the all-reduce helper.
  # Under a module-level lock, reads and increments a global counter and returns a string made of
  # the letter c followed by the previous counter value. Takes no inputs. Side effect: mutates
  # the module-global counter; thread safe.
  global _shared_name_counter

  with _module_lock:
    val = _shared_name_counter
    _shared_name_counter += 1
  return 'c%s' % val


def _check_device(tensor, expected=None):
  # [FUNCTION DOC]
  # HELPER: validates device placement of tensors for the NCCL ops in this module.
  # Inputs: a tensor and an optional expected device string. Raises a value error if the tensor
  # has no canonical device name, or if an expected device is given and differs from the tensor
  # device. Returns nothing.
  if not device.canonical_name(tensor.device):
    raise ValueError(f'Device assignment for tensor={tensor} required for nccl '
                     'collective ops')
  if expected and expected != tensor.device:
    raise ValueError(f'Expected device {expected}, got {tensor.device} for '
                     f'tensor={tensor}.')
