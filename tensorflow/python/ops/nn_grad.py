# [MODULE DOC]
# Registers gradient functions for neural-network ops defined in nn_ops / gen_nn_ops: convolutions
# (2D, 3D, depthwise, dilation) and their backprop ops (for higher-order gradients), pooling
# (average, max, fractional, 3D, with argmax and their grad-of-grad variants), softmax and
# log-softmax, softmax cross entropy, bias add, activations (relu, relu6, leaky relu, elu, selu,
# softplus, softsign), local response normalization, batch norms, L2 loss, top-k, approximate
# top-k, nth element and isotonic regression.
# Each function is attached to an op type name through the gradient registry at import time
# (the only import-time side effect). A gradient function receives the forward op and the
# gradients flowing into its outputs, and returns one gradient (or nothing) per op input.
# Mostly implemented by calling the generated gen_nn_ops kernels directly (faster in eager mode).
# Private helpers are used only inside this module. Not a helper module itself; it is loaded by
# the autodiff machinery.
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
"""Gradients for operators defined in nn_ops.py."""

import functools
import itertools
import operator

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops


@ops.RegisterGradient("Conv2DBackpropInput")
def _Conv2DBackpropInputGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the Conv2DBackpropInput (transposed convolution) op, used for second-order
  # gradients. Inputs: op, the forward op (its attributes dilations, strides, padding, explicit
  # paddings, cudnn flag and data format are read); grad, gradient with respect to the op output.
  # Returns a list of three: nothing for the input-sizes input, the filter gradient (computed by
  # conv2d backprop-filter from grad, the filter shape and the third input), and the gradient for
  # the third input (a plain conv2d of grad with the filter). Uses the generated kernels directly
  # for eager-mode performance. No side effects.
  """The derivatives for deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
  # We call the gen_nn_ops backprop functions instead of nn_ops backprop
  # functions for performance reasons in Eager mode. See _Conv2DGrad.
  return [
      None,
      gen_nn_ops.conv2d_backprop_filter(
          grad,
          array_ops.shape(op.inputs[1]),
          op.inputs[2],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          use_cudnn_on_gpu=op.get_attr("use_cudnn_on_gpu"),
          data_format=op.get_attr("data_format").decode()),
      gen_nn_ops.conv2d(
          grad,
          op.inputs[1],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          use_cudnn_on_gpu=op.get_attr("use_cudnn_on_gpu"),
          data_format=op.get_attr("data_format").decode())
  ]


@ops.RegisterGradient("Conv2DBackpropFilter")
def _Conv2DBackpropFilterGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the Conv2DBackpropFilter op (for second-order gradients). Inputs: op, the forward
  # op whose attributes (dilations, strides, padding, explicit paddings, cudnn flag, data format)
  # are reused; grad, gradient with respect to the filter-gradient output.
  # Returns a list of three: gradient for the original input (conv2d backprop-input using the
  # input shape, grad and the third input), nothing for the filter-sizes input, and the gradient
  # for the third input (a conv2d of the original input with grad). No side effects.
  # We call the gen_nn_ops backprop functions instead of nn_ops backprop
  # functions for performance reasons in Eager mode. See _Conv2DGrad.
  return [
      gen_nn_ops.conv2d_backprop_input(
          array_ops.shape(op.inputs[0]),
          grad,
          op.inputs[2],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          use_cudnn_on_gpu=op.get_attr("use_cudnn_on_gpu"),
          data_format=op.get_attr("data_format").decode()), None,
      gen_nn_ops.conv2d(
          op.inputs[0],
          grad,
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          use_cudnn_on_gpu=op.get_attr("use_cudnn_on_gpu"),
          data_format=op.get_attr("data_format").decode())
  ]


@ops.RegisterGradient("DepthwiseConv2dNativeBackpropInput")
def _DepthwiseConv2dNativeBackpropInputGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the depthwise conv2d backprop-input op (second-order gradients). Inputs: op, the
  # forward op (dilations, strides, padding, explicit paddings, data format attributes are read);
  # grad, gradient with respect to the op output.
  # Returns a list of three: nothing for the input-sizes input, the filter gradient (depthwise
  # backprop-filter of grad against the third input) and the gradient for the third input (a
  # depthwise conv2d of grad with the filter). No side effects.
  """The derivatives for deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
  return [
      None,
      gen_nn_ops.depthwise_conv2d_native_backprop_filter(
          grad,
          array_ops.shape(op.inputs[1]),
          op.inputs[2],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          data_format=op.get_attr("data_format")),
      gen_nn_ops.depthwise_conv2d_native(
          grad,
          op.inputs[1],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          data_format=op.get_attr("data_format"))
  ]


@ops.RegisterGradient("DepthwiseConv2dNativeBackpropFilter")
def _DepthwiseConv2dNativeBackpropFilterGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the depthwise conv2d backprop-filter op (second-order gradients). Inputs: op, the
  # forward op whose attributes are reused; grad, gradient with respect to its output (filter
  # shaped). Returns a list of three: gradient for the original input (depthwise backprop-input),
  # nothing for the filter-sizes input, and gradient for the third input (depthwise conv2d of the
  # original input with grad). No side effects.
  return [
      gen_nn_ops.depthwise_conv2d_native_backprop_input(
          array_ops.shape(op.inputs[0]),
          grad,
          op.inputs[2],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          data_format=op.get_attr("data_format")), None,
      gen_nn_ops.depthwise_conv2d_native(
          op.inputs[0],
          grad,
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          data_format=op.get_attr("data_format"))
  ]


@ops.RegisterGradient("Conv3D")
def _Conv3DGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the 3D convolution op. Inputs: op, the forward Conv3D op (dilations, strides,
  # padding and data format attributes are read; the data format is decoded from bytes); grad,
  # gradient with respect to the output. Returns a list of two: the gradient for the input
  # (conv3d backprop-input v2, using the input shape) and for the filter (conv3d backprop-filter
  # v2, using the filter shape). Shapes are obtained together via a shape-of-many op.
  data_format = op.get_attr("data_format").decode()
  shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
  return [
      gen_nn_ops.conv3d_backprop_input_v2(
          shape_0,
          op.inputs[1],
          grad,
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=data_format),
      gen_nn_ops.conv3d_backprop_filter_v2(
          op.inputs[0],
          shape_1,
          grad,
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=data_format,
      ),
  ]


@ops.RegisterGradient("Conv3DBackpropInputV2")
def _Conv3DBackpropInputGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the Conv3DBackpropInputV2 op (second-order gradients). Inputs: op, the forward
  # op (dilations, strides, padding, data format attributes read); grad, gradient with respect to
  # its output. Returns a list of three: nothing for the input-sizes input, the filter gradient
  # (conv3d backprop-filter v2) and the gradient for the third input (a conv3d of grad with the
  # filter).
  data_format = op.get_attr("data_format").decode()
  return [
      None,
      gen_nn_ops.conv3d_backprop_filter_v2(
          grad,
          array_ops.shape(op.inputs[1]),
          op.inputs[2],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=data_format),
      gen_nn_ops.conv3d(
          grad,
          op.inputs[1],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=data_format)
  ]


@ops.RegisterGradient("Conv3DBackpropFilterV2")
def _Conv3DBackpropFilterGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the Conv3DBackpropFilterV2 op (second-order gradients). Inputs: op, the forward
  # op whose attributes (dilations, strides, padding, data format) are reused; grad, gradient
  # with respect to its output. Returns a list of three: the gradient for the original input
  # (conv3d backprop-input v2), nothing for the filter-sizes input, and the gradient for the
  # third input (conv3d of the original input with grad).
  data_format = op.get_attr("data_format").decode()
  return [
      gen_nn_ops.conv3d_backprop_input_v2(
          array_ops.shape(op.inputs[0]),
          grad,
          op.inputs[2],
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=data_format), None,
      gen_nn_ops.conv3d(
          op.inputs[0],
          grad,
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=data_format)
  ]


@ops.RegisterGradient("AvgPool3D")
def _AvgPool3DGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of 3D average pooling. Inputs: op, the forward op (ksize, strides, padding, data
  # format attributes read); grad, gradient with respect to the pooled output. Returns the single
  # gradient tensor for the input, computed by the avg-pool3d-grad kernel using the input shape.
  return gen_nn_ops.avg_pool3d_grad(
      array_ops.shape(op.inputs[0]),
      grad,
      ksize=op.get_attr("ksize"),
      strides=op.get_attr("strides"),
      padding=op.get_attr("padding"),
      data_format=op.get_attr("data_format").decode())


@ops.RegisterGradient("AvgPool3DGrad")
def _AvgPool3DGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the AvgPool3DGrad op (second-order). Inputs: op, the forward grad op; grad,
  # gradient with respect to its output. Returns a pair: a stopped-gradient copy of the first
  # input (the original shape input, not differentiable) and the result of average pooling grad
  # with the same ksize, strides, padding and data format.
  return (array_ops.stop_gradient(op.inputs[0]),
          gen_nn_ops.avg_pool3d(
              grad,
              op.get_attr("ksize"),
              op.get_attr("strides"),
              op.get_attr("padding"),
              data_format=op.get_attr("data_format").decode()))


@ops.RegisterGradient("MaxPool3D")
def _MaxPool3DGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of 3D max pooling. Inputs: op, the forward op (ksize, strides, padding, data format
  # read); grad, gradient with respect to the pooled output. Returns the input gradient computed
  # by the max-pool3d-grad kernel from the original input, the pooled output and grad.
  return gen_nn_ops.max_pool3d_grad(
      op.inputs[0],
      op.outputs[0],
      grad,
      ksize=op.get_attr("ksize"),
      strides=op.get_attr("strides"),
      padding=op.get_attr("padding"),
      data_format=op.get_attr("data_format").decode())


@ops.RegisterGradient("MaxPool3DGrad")
def _MaxPool3DGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the MaxPool3DGrad op (second-order). Inputs: op, the forward grad op with three
  # inputs (original input, pooled output, upstream gradient); grad, gradient with respect to its
  # output. Returns a triple: zeros shaped like the first two inputs and the max-pool3d
  # grad-of-grad result for the third input, using the op's ksize, strides, padding, data format.
  return (array_ops.zeros_like(op.inputs[0]),
          array_ops.zeros_like(op.inputs[1]),
          gen_nn_ops.max_pool3d_grad_grad(
              op.inputs[0],
              op.inputs[1],
              grad,
              op.get_attr("ksize"),
              op.get_attr("strides"),
              padding=op.get_attr("padding"),
              data_format=op.get_attr("data_format").decode()))


@ops.RegisterGradient("MaxPool3DGradGrad")
def _MaxPool3DGradGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the MaxPool3DGradGrad op (third-order). Inputs: op, the forward op with inputs
  # (original input, pooled output, upstream gradient); grad, gradient with respect to its output.
  # Returns a triple: zeros shaped like the first two inputs and the max-pool3d-grad result for
  # the third input, using the op's ksize, strides, padding and data format.
  return (array_ops.zeros_like(op.inputs[0]),
          array_ops.zeros_like(op.inputs[1]),
          gen_nn_ops.max_pool3d_grad(
              op.inputs[0],
              op.inputs[1],
              grad,
              op.get_attr("ksize"),
              op.get_attr("strides"),
              padding=op.get_attr("padding"),
              data_format=op.get_attr("data_format").decode()))


@ops.RegisterGradient("Softmax")
def _SoftmaxGrad(op: ops.Operation, grad_softmax):
  # [FUNCTION DOC]
  # Gradient of softmax. Inputs: op, the Softmax op (its output is reused); grad_softmax,
  # gradient with respect to the softmax output. Computes, along the last axis, the incoming
  # gradient minus the sum of (gradient times softmax), multiplied by softmax, and returns that
  # single tensor with the input's shape. Works for any rank, softmax taken over the last axis.
  """The derivative of the softmax nonlinearity.

  We assume that probs is of shape [batch_size * dim]
  The formula for dsoftmax / dx = (diag(softmax) - softmax * softmax').
  This matrix is diagonal minus a rank one matrix, so it is easy to implement
  as follows:

    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

  Args:
     op: the Softmax op.
     grad_softmax:  the tensor representing the gradient w.r.t. the softmax
       output.

  Returns:
     gradient w.r.t the input to the softmax

  """
  softmax = op.outputs[0]
  sum_channels = math_ops.reduce_sum(grad_softmax * softmax, -1, keepdims=True)
  return (grad_softmax - sum_channels) * softmax


@ops.RegisterGradient("LogSoftmax")
def _LogSoftmaxGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of log-softmax. Inputs: op, the LogSoftmax op (softmax is recovered by taking the
  # exponential of its output); grad, gradient with respect to the output. Returns grad minus the
  # last-axis sum of grad times the softmax, a single tensor shaped like the input.
  """The gradient for log_softmax.

      log_softmax = input - log(sum(exp(input))
      dlog_softmax/dinput = diag - softmax(input)

  Args:
    op: The log softmax op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
  softmax = math_ops.exp(op.outputs[0])
  return grad - math_ops.reduce_sum(grad, -1, keepdims=True) * softmax


@ops.RegisterGradient("BiasAdd")
def _BiasAddGrad(op: ops.Operation, received_grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the bias gradient is the sum over the batch (first)
  # dimension; the code actually delegates to the bias-add-grad kernel, which sums over every
  # dimension except the channel dimension chosen by the data format attribute.
  # Gradient of bias addition. Inputs: op, the BiasAdd op; received_grad, gradient with respect to
  # the output. Reads the data format attribute, falling back to nothing if the op lacks it.
  # Returns a pair: the received gradient unchanged for the tensor input, and the bias gradient.
  """Return the gradients for the 2 inputs of bias_op.

  The first input of unused_bias_op is the tensor t, and its gradient is
  just the gradient the unused_bias_op received.

  The second input of unused_bias_op is the bias vector which has one fewer
  dimension than "received_grad" (the batch dimension.)  Its gradient is the
  received gradient Summed on the batch dimension, which is the first dimension.

  Args:
    op: The BiasOp for which we need to generate gradients.
    received_grad: Tensor.  The gradients passed to the BiasOp.

  Returns:
    Two tensors, the first one for the "tensor" input of the BiasOp,
    the second one for the "bias" input of the BiasOp.
  """
  try:
    data_format = op.get_attr("data_format")
  except ValueError:
    data_format = None
  return (received_grad,
          gen_nn_ops.bias_add_grad(
              out_backprop=received_grad, data_format=data_format))


@ops.RegisterGradient("BiasAddGrad")
def _BiasAddGradGrad(op: ops.Operation, received_grad):
  # [FUNCTION DOC]
  # Gradient of the BiasAddGrad op (second-order). Inputs: op, the BiasAddGrad op (its input
  # supplies the full shape; the data format attribute is read, defaulting to nothing when
  # absent); received_grad, a bias-sized gradient. Reshapes the bias-sized gradient so the bias
  # axis is the channel axis (second axis for channels-first format, otherwise last axis) with
  # ones elsewhere, then tiles it across the other dimensions to the input's full shape.
  # Returns that single tensor.
  """Gradient for the BiasAddGrad op.

  Args:
    op: BiasAddGrad op for which we are calculating gradients.
    received_grad: The gradients passed to the BiasAddGrad op.

  Returns:
    A single gradient Tensor for the input to BiasAddGrad (which
    is the gradient of the bias term in BiasAdd)
  """

  try:
    data_format = op.get_attr("data_format")
  except ValueError:
    data_format = None

  shape = array_ops.shape(op.inputs[0])
  bias_shape = array_ops.shape(received_grad)

  if data_format == b"NCHW":
    expanded_shape = array_ops.concat([
        array_ops.ones_like(shape[:1]), bias_shape,
        array_ops.ones_like(shape[2:])
    ], 0)
    tile_mults = array_ops.concat([shape[:1], [1], shape[2:]], 0)
  else:
    expanded_shape = array_ops.concat(
        [array_ops.ones_like(shape[:-1]), bias_shape], 0)
    tile_mults = array_ops.concat([shape[:-1], [1]], 0)

  expanded_grad = array_ops.reshape(received_grad, expanded_shape)
  return array_ops.tile(expanded_grad, tile_mults)


@ops.RegisterGradient("BiasAddV1")
def _BiasAddGradV1(unused_bias_op: ops.Operation, received_grad):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the bias gradient sums over the batch (first) dimension, but
  # the code sums over all dimensions except the last (channels-last assumed).
  # Gradient of the legacy BiasAddV1 op. Inputs: the op (unused), and received_grad, the gradient
  # with respect to the output. Returns a pair: received_grad for the tensor input, and the sum of
  # received_grad over all axes but the last for the bias input.
  """Return the gradients for the 2 inputs of bias_op.

  The first input of unused_bias_op is the tensor t, and its gradient is
  just the gradient the unused_bias_op received.

  The second input of unused_bias_op is the bias vector which has one fewer
  dimension than "received_grad" (the batch dimension.)  Its gradient is the
  received gradient Summed on the batch dimension, which is the first dimension.

  Args:
    unused_bias_op: The BiasOp for which we need to generate gradients.
    received_grad: Tensor.  The gradients passed to the BiasOp.

  Returns:
    Two tensors, the first one for the "tensor" input of the BiasOp,
    the second one for the "bias" input of the BiasOp.
  """
  reduction_dim_tensor = math_ops.range(array_ops.rank(received_grad) - 1)
  return (received_grad, math_ops.reduce_sum(received_grad,
                                             reduction_dim_tensor))


@ops.RegisterGradient("Relu")
def _ReluGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of relu: applies the relu-grad kernel to the incoming gradient and the op's output.
  # Returns one tensor shaped like the input.
  return gen_nn_ops.relu_grad(grad, op.outputs[0])


@ops.RegisterGradient("EluGrad")
def _EluGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the EluGrad op (second-order). Inputs: op, with inputs (upstream gradient, elu
  # activation value); grad, gradient with respect to the op output. Returns a pair: the elu-grad
  # of grad at the activation (for the upstream gradient input), and for the activation input the
  # product of grad and the upstream gradient where the activation is negative, zero elsewhere.
  elu_x = op.inputs[1]
  return (gen_nn_ops.elu_grad(grad, elu_x),
          array_ops.where(
              elu_x < 0, grad * op.inputs[0], array_ops.zeros_like(elu_x)))


@ops.RegisterGradient("SeluGrad")
def _SeluGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the SeluGrad op (second-order). Inputs: op, with inputs (upstream gradient, selu
  # activation value); grad, gradient with respect to the op output. Returns a pair: the selu-grad
  # of grad at the activation, and for the activation input the product of grad and the upstream
  # gradient where the activation is negative, zero elsewhere.
  selu_x = op.inputs[1]
  return (gen_nn_ops.selu_grad(grad, selu_x),
          array_ops.where(
              selu_x < 0., grad * op.inputs[0], array_ops.zeros_like(selu_x)))


@ops.RegisterGradient("Relu6")
def _Relu6Grad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of relu6: applies the relu6-grad kernel to the incoming gradient and the op's
  # output. Returns one tensor shaped like the input.
  return gen_nn_ops.relu6_grad(grad, op.outputs[0])


@ops.RegisterGradient("Relu6Grad")
def _Relu6GradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the Relu6Grad op (second-order). Inputs: op, with inputs (upstream gradient,
  # features); grad, gradient with respect to its output. Returns a pair: relu6-grad of grad at
  # the features for the upstream-gradient input, and zeros shaped like the features.
  x = op.inputs[1]
  return (gen_nn_ops.relu6_grad(grad, x), array_ops.zeros_like(x))


@ops.RegisterGradient("LeakyRelu")
def _LeakyReluGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of leaky relu. Reads the alpha (negative slope) attribute from the op and applies the
  # leaky-relu-grad kernel to the incoming gradient and the op's input. Returns one tensor.
  x = op.inputs[0]
  alpha = op.get_attr("alpha")
  return gen_nn_ops.leaky_relu_grad(grad, x, alpha=alpha)


@ops.RegisterGradient("LeakyReluGrad")
def _LeakyReluGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the LeakyReluGrad op (second-order). Inputs: op, with inputs (upstream gradient,
  # features) and an alpha attribute; grad, gradient with respect to its output. Returns a pair:
  # leaky-relu-grad of grad at the features, and zeros shaped like the features.
  x = op.inputs[1]
  alpha = op.get_attr("alpha")
  return (gen_nn_ops.leaky_relu_grad(grad, x,
                                     alpha=alpha), array_ops.zeros_like(x))


@ops.RegisterGradient("Elu")
def _EluGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of elu: applies the elu-grad kernel to the incoming gradient and the op's output.
  # Returns one tensor shaped like the input.
  return gen_nn_ops.elu_grad(grad, op.outputs[0])


@ops.RegisterGradient("Selu")
def _SeluGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of selu: applies the selu-grad kernel to the incoming gradient and the op's output.
  # Returns one tensor shaped like the input.
  return gen_nn_ops.selu_grad(grad, op.outputs[0])


@ops.RegisterGradient("Softplus")
def _SoftplusGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of softplus: the incoming gradient multiplied elementwise by the sigmoid of the op's
  # input. Returns one tensor shaped like the input.
  return grad * math_ops.sigmoid(op.inputs[0])


@ops.RegisterGradient("SoftplusGrad")
def _SoftplusGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the SoftplusGrad op (second-order). Inputs: op, with inputs (upstream gradient
  # dy, features x); grad, gradient with respect to its output. Returns a pair: the softplus-grad
  # of grad at x (for dy), and grad times dy divided by (exp(-x) + 2 + exp(x)) (for x). The
  # computation is wrapped in a control dependency on grad so grad is evaluated first.
  # Let:
  #   y = tf.nn.softplus(x)
  #   dx = gen_nn_ops.softplus_grad(dy, x) = dy / (1 + exp(-x))
  # This op computes (ddy, d2x) from op.inputs == [dy, x] and grad == ddx.
  dy, x = op.inputs
  with ops.control_dependencies([grad]):
    ddy = gen_nn_ops.softplus_grad(grad, x)
    d2x = grad * dy / (math_ops.exp(-x) + 2.0 + math_ops.exp(x))
    return (ddy, d2x)


@ops.RegisterGradient("Softsign")
def _SoftsignGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of softsign: applies the softsign-grad kernel to the incoming gradient and the op's
  # input. Returns one tensor shaped like the input.
  return gen_nn_ops.softsign_grad(grad, op.inputs[0])


@ops.RegisterGradient("ReluGrad")
def _ReluGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the ReluGrad op (second-order). Inputs: op, with inputs (upstream gradient,
  # features); grad, gradient with respect to its output. Returns a pair: relu-grad of grad at the
  # features, and zeros shaped like the features.
  x = op.inputs[1]
  return (gen_nn_ops.relu_grad(grad, x), array_ops.zeros_like(x))


def _BroadcastMul(vec, mat):
  # [FUNCTION DOC]
  # HELPER: used by the softmax cross-entropy gradient functions in this module.
  # Multiplies a one-dimensional tensor (one value per row) with a two-dimensional tensor by
  # first adding a trailing axis to the vector so it broadcasts across columns. Returns the
  # product with the matrix's shape. No side effects.
  """Multiply after broadcasting vec to match dimensions of mat.

  Args:
    vec: A 1-D tensor of dimension [D0]
    mat: A 2-D tensor of dimension [D0, D1]

  Returns:
    A tensor of dimension [D0, D1], the result of vec * mat
  """
  # Reshape vec to [D0, 1]
  vec = array_ops.expand_dims(vec, -1)
  return vec * mat


@ops.RegisterGradient("SoftmaxCrossEntropyWithLogits")
def _SoftmaxCrossEntropyWithLogitsGrad(op: ops.Operation, grad_loss, grad_grad):
  # [FUNCTION DOC]
  # Gradient of softmax cross entropy with logits. Inputs: op (logits and labels as inputs; loss
  # and precomputed backprop as outputs); grad_loss, gradient for the loss output; grad_grad,
  # gradient for the backprop output (may be nothing or a marked all-zeros tensor, in which case
  # the second-order term is skipped). The logits gradient is grad_loss scaled backprop, plus,
  # when grad_grad is present, a correction using the softmax of the logits. The labels gradient
  # is grad_loss times the negative log-softmax of the logits. Returns a pair (logits, labels).
  """Gradient function for SoftmaxCrossEntropyWithLogits."""
  # grad_loss is the backprop for cost, and we multiply it with the gradients
  # (which is output[1])
  # grad_grad is the backprop for softmax gradient.
  #
  # Second derivative is just softmax derivative w.r.t. logits.
  softmax_grad = op.outputs[1]
  grad = _BroadcastMul(grad_loss, softmax_grad)

  logits = op.inputs[0]
  if (grad_grad is not None and
      not getattr(grad_grad, "_is_zeros_tensor", False)):
    softmax = gen_nn_ops.softmax(logits)

    grad += ((grad_grad - array_ops.squeeze(
        math_ops.matmul(
            array_ops.expand_dims(grad_grad, 1),
            array_ops.expand_dims(softmax, 2)),
        axis=1)) * softmax)

  return grad, _BroadcastMul(grad_loss, -gen_nn_ops.log_softmax(logits))  # pylint: disable=invalid-unary-operand-type


@ops.RegisterGradient("SparseSoftmaxCrossEntropyWithLogits")
def _SparseSoftmaxCrossEntropyWithLogitsGrad(op: ops.Operation,
                                             grad_loss,
                                             grad_grad):
  # [FUNCTION DOC]
  # Gradient of sparse softmax cross entropy with logits. Inputs: op (logits and integer labels;
  # loss and precomputed backprop as outputs); grad_loss, gradient for the loss; grad_grad,
  # gradient for the backprop output (nothing or a marked all-zeros tensor skips the second-order
  # term). Returns a pair: the logits gradient (grad_loss scaled backprop, plus a softmax-based
  # correction when grad_grad is given) and nothing for the labels, which are not differentiable.
  """Gradient function for SparseSoftmaxCrossEntropyWithLogits."""
  # grad_loss is the backprop for cost, and we multiply it with the gradients
  # (which is output[1])
  # grad_grad is the backprop for softmax gradient.
  # There is no gradient for the labels
  #
  # Second derivative is just softmax derivative w.r.t. logits.
  softmax_grad = op.outputs[1]
  grad = _BroadcastMul(grad_loss, softmax_grad)

  logits = op.inputs[0]
  if (grad_grad is not None and
      not getattr(grad_grad, "_is_zeros_tensor", False)):
    softmax = gen_nn_ops.softmax(logits)

    grad += ((grad_grad - array_ops.squeeze(
        math_ops.matmul(
            array_ops.expand_dims(grad_grad, 1),
            array_ops.expand_dims(softmax, 2)),
        axis=1)) * softmax)

  return grad, None


@ops.RegisterGradient("Conv2D")
def _Conv2DGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of 2D convolution. Inputs: op, the Conv2D op (dilations, strides, padding, explicit
  # paddings, cudnn flag and data format attributes read); grad, gradient with respect to the
  # output. Returns a list of two: the input gradient (conv2d backprop-input using the input
  # shape) and the filter gradient (conv2d backprop-filter using the filter shape). Uses the
  # generated kernels directly because they accept explicit paddings and avoid eager overhead.
  """Gradient function for Conv2D."""
  dilations = op.get_attr("dilations")
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")
  explicit_paddings = op.get_attr("explicit_paddings")
  use_cudnn_on_gpu = op.get_attr("use_cudnn_on_gpu")
  data_format = op.get_attr("data_format")
  shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])

  # We call the gen_nn_ops backprop functions instead of nn_ops backprop
  # functions for performance reasons in Eager mode. gen_nn_ops functions take a
  # `explicit_paddings` parameter, but nn_ops functions do not. So if we were
  # to use the nn_ops functions, we would have to convert `padding` and
  # `explicit_paddings` into a single `padding` parameter, increasing overhead
  # in Eager mode.
  return [
      gen_nn_ops.conv2d_backprop_input(
          shape_0,
          op.inputs[1],
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          explicit_paddings=explicit_paddings,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format),
      gen_nn_ops.conv2d_backprop_filter(
          op.inputs[0],
          shape_1,
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          explicit_paddings=explicit_paddings,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format)
  ]


@ops.RegisterGradient("DepthwiseConv2dNative")
def _DepthwiseConv2dNativeGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of depthwise 2D convolution. Inputs: op (attributes dilations, strides, padding,
  # explicit paddings, data format read); grad, gradient with respect to the output. Returns a
  # list of two: the input gradient (depthwise backprop-input) and the filter gradient
  # (depthwise backprop-filter).
  return [
      gen_nn_ops.depthwise_conv2d_native_backprop_input(
          array_ops.shape(op.inputs[0]),
          op.inputs[1],
          grad,
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          data_format=op.get_attr("data_format")),
      gen_nn_ops.depthwise_conv2d_native_backprop_filter(
          op.inputs[0],
          array_ops.shape(op.inputs[1]),
          grad,
          dilations=op.get_attr("dilations"),
          strides=op.get_attr("strides"),
          padding=op.get_attr("padding"),
          explicit_paddings=op.get_attr("explicit_paddings"),
          data_format=op.get_attr("data_format"))
  ]


@ops.RegisterGradient("Dilation2D")
def _Dilation2DGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of 2D morphological dilation. Inputs: op (strides, rates and padding read); grad,
  # gradient with respect to the output. Returns a list of two: gradient for the input and for
  # the filter, each from its dilation2d backprop kernel.
  return [
      gen_nn_ops.dilation2d_backprop_input(op.inputs[0], op.inputs[1], grad,
                                           op.get_attr("strides"),
                                           op.get_attr("rates"),
                                           op.get_attr("padding")),
      gen_nn_ops.dilation2d_backprop_filter(op.inputs[0], op.inputs[1], grad,
                                            op.get_attr("strides"),
                                            op.get_attr("rates"),
                                            op.get_attr("padding"))
  ]


@ops.RegisterGradient("LRN")
def _LRNGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of local response normalization. Reads depth radius, bias, alpha and beta from the
  # op and calls the lrn-grad kernel with the incoming gradient, the op input and output.
  # Returns a list with a single input gradient.
  depth_radius = op.get_attr("depth_radius")
  bias = op.get_attr("bias")
  alpha = op.get_attr("alpha")
  beta = op.get_attr("beta")
  return [
      gen_nn_ops.lrn_grad(grad, op.inputs[0], op.outputs[0], depth_radius, bias,
                          alpha, beta)
  ]


@ops.RegisterGradient("AvgPool")
def _AvgPoolGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of 2D average pooling. Uses the input's shape (as 32-bit integers), grad, and the
  # op's ksize, strides, padding and data format with the avg-pool-grad kernel. Returns the input
  # gradient tensor.
  return gen_nn_ops.avg_pool_grad(
      array_ops.shape(op.inputs[0], out_type=dtypes.int32),
      grad,
      op.get_attr("ksize"),
      op.get_attr("strides"),
      op.get_attr("padding"),
      data_format=op.get_attr("data_format"))


@ops.RegisterGradient("AvgPoolGrad")
def _AvgPoolGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the AvgPoolGrad op (second-order). Returns a pair: a stopped-gradient copy of the
  # first input (the shape input, not differentiable) and the average pooling of grad using the
  # op's ksize, strides, padding and data format.
  return (array_ops.stop_gradient(op.inputs[0]),
          gen_nn_ops.avg_pool(
              grad,
              op.get_attr("ksize"),
              op.get_attr("strides"),
              op.get_attr("padding"),
              data_format=op.get_attr("data_format")))


@ops.RegisterGradient("MaxPool")
def _MaxPoolGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of 2D max pooling. Calls the max-pool-grad kernel with the op's input, output, grad,
  # and its ksize, strides, padding, explicit paddings and data format attributes. Returns the
  # input gradient tensor.
  return gen_nn_ops.max_pool_grad(
      op.inputs[0],
      op.outputs[0],
      grad,
      op.get_attr("ksize"),
      op.get_attr("strides"),
      padding=op.get_attr("padding"),
      explicit_paddings=op.get_attr("explicit_paddings"),
      data_format=op.get_attr("data_format"))


@ops.RegisterGradient("MaxPoolV2")
def _MaxPoolGradV2(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of MaxPoolV2, where ksize and strides are tensor inputs (second and third inputs).
  # Returns a triple: the max-pool-grad-v2 result for the data input and nothing for ksize and
  # strides. Padding and data format are read from op attributes.
  ksize = op.inputs[1]
  strides = op.inputs[2]
  return gen_nn_ops.max_pool_grad_v2(
      op.inputs[0],
      op.outputs[0],
      grad,
      ksize,
      strides,
      padding=op.get_attr("padding"),
      data_format=op.get_attr("data_format")), None, None


@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op: ops.Operation, grad, unused_argmax_grad):
  # [FUNCTION DOC]
  # Gradient of max pooling with argmax. Inputs: op; grad, gradient for the pooled values;
  # unused_argmax_grad, gradient for the integer argmax output, discarded. Returns the input
  # gradient from the argmax-based kernel using the op's argmax output, ksize, strides, padding
  # and include-batch-in-index attributes.
  del unused_argmax_grad
  return gen_nn_ops.max_pool_grad_with_argmax(
      op.inputs[0],
      grad,
      op.outputs[1],
      op.get_attr("ksize"),
      op.get_attr("strides"),
      padding=op.get_attr("padding"),
      include_batch_in_index=op.get_attr("include_batch_in_index"),
  )


@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
  # [FUNCTION DOC]
  # Gradient of the MaxPoolGrad op (second-order). The op argument has no type annotation here
  # unlike its siblings. Returns a triple: zeros shaped like the first two inputs and the
  # max-pool grad-of-grad result for the third input, using ksize, strides, padding and data
  # format attributes of the op.
  return (
      array_ops.zeros_like(op.inputs[0]),
      array_ops.zeros_like(op.inputs[1]),
      gen_nn_ops.max_pool_grad_grad(
          op.inputs[0],
          op.inputs[1],
          grad,
          op.get_attr("ksize"),
          op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=op.get_attr("data_format"),
      ),
  )


@ops.RegisterGradient("MaxPoolGradV2")
def _MaxPoolGradGradV2(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the MaxPoolGradV2 op (second-order), where ksize and strides are the fourth and
  # fifth inputs. Returns a five-element tuple: zeros for the first two inputs, the v2 max-pool
  # grad-of-grad result for the third, and nothing for ksize and strides.
  ksize = op.inputs[3]
  strides = op.inputs[4]
  return (
      array_ops.zeros_like(op.inputs[0]),
      array_ops.zeros_like(op.inputs[1]),
      gen_nn_ops.max_pool_grad_grad_v2(
          op.inputs[0],
          op.inputs[1],
          grad,
          ksize,
          strides,
          padding=op.get_attr("padding"),
          data_format=op.get_attr("data_format"),
      ),
      None,
      None,
  )


@ops.RegisterGradient("MaxPoolGradGrad")
def _MaxPoolGradGradGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the MaxPoolGradGrad op (third-order). Returns a triple: zeros for the first two
  # inputs and the max-pool-grad result for the third input, using the op's ksize, strides,
  # padding and data format.
  return (
      array_ops.zeros_like(op.inputs[0]),
      array_ops.zeros_like(op.inputs[1]),
      gen_nn_ops.max_pool_grad(
          op.inputs[0],
          op.inputs[1],
          grad,
          op.get_attr("ksize"),
          op.get_attr("strides"),
          padding=op.get_attr("padding"),
          data_format=op.get_attr("data_format"),
      ),
  )


@ops.RegisterGradient("FractionalMaxPool")
def _FractionalMaxPoolGrad(
    op: ops.Operation, grad_0, unused_grad_1, unused_grad_2
):
  # [FUNCTION DOC]
  # Gradient of fractional max pooling. Inputs: op (three outputs: pooled values, row and column
  # pooling sequences); grad_0, gradient for the pooled values; two unused gradients for the
  # sequences, ignored. Returns the input gradient from the fractional-max-pool-grad kernel using
  # the op's input, output, sequences and overlapping attribute.
  """Returns gradient for FractionalMaxPool.

  Since FractionalMaxPool has three outputs, there are three gradients passed in
  for each of the outputs. Only the first one is useful, the other two gradients
  are empty.

  Args:
    op: The FractionalMaxPoolOp.
    grad_0: Gradient with respect to op.outputs[0]
    unused_grad_1: Gradient with respect to op.outputs[1]/row_seq. It is empty.
    unused_grad_2: Gradient with respect to op.outputs[2]/col_seq. It is empty.

  Returns:
    Input backprop for FractionalMaxPool op.
  """
  return gen_nn_ops.fractional_max_pool_grad(
      op.inputs[0],
      op.outputs[0],
      grad_0,
      op.outputs[1],
      op.outputs[2],
      op.get_attr("overlapping"),
  )


@ops.RegisterGradient("FractionalAvgPool")
def _FractionalAvgPoolGrad(
    op: ops.Operation, grad_0, unused_grad_1, unused_grad_2
):
  # [FUNCTION DOC]
  # Gradient of fractional average pooling. Inputs: op (three outputs: pooled values, row and
  # column pooling sequences); grad_0, gradient for the pooled values; two unused gradients for
  # the sequences, ignored. Returns the input gradient from the fractional-avg-pool-grad kernel,
  # given the input's static shape, grad_0, the sequences and the overlapping attribute.
  # Requires the input shape to be statically known.
  """Returns gradient for FractionalAvgPool.

  Since FractionalAvgPool has three outputs, there are three gradients passed in
  for each of the outputs. Only the first one is useful, the other two gradients
  are empty.

  Args:
    op: The FractionalAvgPoolOp.
    grad_0: Gradient with respect to op.outputs[0]
    unused_grad_1: Gradient with respect to op.outputs[1]/row_seq. It is empty.
    unused_grad_2: Gradient with respect to op.outputs[2]/col_seq. It is empty.

  Returns:
    Input backprop for FractionalAvgPool op.
  """
  return gen_nn_ops.fractional_avg_pool_grad(op.inputs[0].get_shape(), grad_0,
                                             op.outputs[1], op.outputs[2],
                                             op.get_attr("overlapping"))


@ops.RegisterGradient("BatchNormWithGlobalNormalization")
def _BatchNormWithGlobalNormalizationGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the legacy batch normalization with global normalization op. Inputs: op (five
  # inputs: input, mean, variance, beta, gamma; variance epsilon and scale-after-normalization
  # attributes read); grad, gradient for the output. Calls the dedicated grad kernel and returns
  # five gradients in input order: input, mean, variance, beta, gamma. Per the docstring mean and
  # variance gradients are produced by the kernel, not stopped.
  """Return the gradients for the 5 inputs of BatchNormWithGlobalNormalization.

  We do not backprop anything for the mean and var intentionally as they are
  not being trained with backprop in the operation.

  Args:
    op: The BatchNormOp for which we need to generate gradients.
    grad: Tensor.  The gradients passed to the BatchNormOp.

  Returns:
    dx: Backprop for input, which is (grad * (g * rsqrt(v + epsilon)))
    dm: Backprop for mean, which is
        sum_over_rest(grad * g) * (-1 / rsqrt(v + epsilon))
    dv: Backprop for variance, which is
        sum_over_rest(grad * g * (x - m)) * (-1/2) * (v + epsilon) ^ (-3/2)
    db: Backprop for beta, which is grad reduced in all except the
        last dimension.
    dg: Backprop for gamma, which is (grad * ((x - m) * rsqrt(v + epsilon)))
  """
  dx, dm, dv, db, dg = gen_nn_ops.batch_norm_with_global_normalization_grad(
      op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[4], grad,
      op.get_attr("variance_epsilon"), op.get_attr("scale_after_normalization"))
  return dx, dm, dv, db, dg


def _BaseFusedBatchNormGrad(op: ops.Operation, version, *grad):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the FusedBatchNorm, V2 and V3 gradient functions.
  # KNOWN ISSUE: the docstring says it returns gradients for 3 inputs, but it returns five values
  # (x, scale, offset gradients, plus two nothing values for population mean and variance).
  # Inputs: op (fused batch norm op; epsilon, data format and training-flag attributes read);
  # version, integer 0, 1 or 2 selecting grad kernel v1, v2 or v3 (any value other than 1 or 2
  # picks the original); grad, variable list of output gradients with the first being for y.
  # In training mode calls the kernel with the op's saved reserve-space outputs (a third reserve
  # space for version 2). In inference mode uses the population mean and variance inputs,
  # transposing channels-first (4-D or 5-D) tensors to channels-last before the kernel and
  # transposing dx back afterwards. Returns (dx, dscale, doffset, nothing, nothing).
  """Return the gradients for the 3 inputs of BatchNorm.

  Args:
    op: The BatchNormOp for which we need to compute gradients.
    version: Integer indicating which version to use of the fused batch
      norm gradient.
    *grad: An argument list for tensors of gradients wrt the outputs
      with grad[0] as grad_y.

  Returns:
    grad_x: gradient for x, which is scale * rsqrt(variance + epsilon) *
            [grad_y - mean(grad_y) - (x - mean(x)) *
            mean(grad_y * (x - mean(x))) / (variance + epsilon)]
            in training mode; grad_y * scale * rsqrt(pop_variance + epsilon)
            in freeze mode.

    grad_scale: gradient for scale, which is sum(grad_y * (x - mean(x)) *
                rsqrt(variance + epsilon)) in training mode;
                sum(grad_y * (x - pop_mean) * rsqrt(pop_variance + epsilon))
                in freeze mode.

    grad_offset: gradient for offset, which is sum(grad_y) in training mode;
                 sum(grad_y) in freeze mode.
  """
  x = op.inputs[0]
  grad_y = grad[0]
  scale = op.inputs[1]
  epsilon = op.get_attr("epsilon")
  data_format = op.get_attr("data_format")
  is_training = op.get_attr("is_training")
  if version == 2:
    grad_fun = gen_nn_ops.fused_batch_norm_grad_v3
  elif version == 1:
    grad_fun = gen_nn_ops.fused_batch_norm_grad_v2
  else:
    grad_fun = gen_nn_ops.fused_batch_norm_grad
  if is_training:
    args = {
        "y_backprop": grad_y,
        "x": x,
        "scale": scale,
        "reserve_space_1": op.outputs[3],
        "reserve_space_2": op.outputs[4],
        "epsilon": epsilon,
        "data_format": data_format,
        "is_training": is_training
    }
    if version == 2:
      args["reserve_space_3"] = op.outputs[5]
    dx, dscale, doffset, _, _ = grad_fun(**args)
  else:
    pop_mean = op.inputs[3]
    pop_var = op.inputs[4]
    if data_format == b"NCHW":
      x = array_ops.transpose(x, [0, 2, 3, 1])
      grad_y = array_ops.transpose(grad_y, [0, 2, 3, 1])
    elif data_format == b"NCDHW":
      x = array_ops.transpose(x, [0, 2, 3, 4, 1])
      grad_y = array_ops.transpose(grad_y, [0, 2, 3, 4, 1])
    target_data_format = ("NHWC" if data_format in (b"NCHW",
                                                    b"NHWC") else "NDHWC")
    args = {
        "y_backprop": grad_y,
        "x": x,
        "scale": scale,
        "reserve_space_1": pop_mean,
        "reserve_space_2": pop_var,
        "epsilon": epsilon,
        "data_format": target_data_format,
        "is_training": is_training
    }
    if version == 2:
      args["reserve_space_3"] = op.outputs[5]
    dx, dscale, doffset, _, _ = grad_fun(**args)
    if data_format == b"NCHW":
      dx = array_ops.transpose(dx, [0, 3, 1, 2])
    elif data_format == b"NCDHW":
      dx = array_ops.transpose(dx, [0, 4, 1, 2, 3])
  return dx, dscale, doffset, None, None


@ops.RegisterGradient("FusedBatchNorm")
def _FusedBatchNormGrad(op: ops.Operation, *grad):
  # [FUNCTION DOC]
  # Gradient of FusedBatchNorm (version 0). Forwards the op and all output gradients to the shared
  # fused batch norm gradient helper and returns its five values.
  return _BaseFusedBatchNormGrad(op, 0, *grad)


@ops.RegisterGradient("FusedBatchNormV2")
def _FusedBatchNormV2Grad(op: ops.Operation, *grad):
  # [FUNCTION DOC]
  # Gradient of FusedBatchNormV2. Forwards to the shared fused batch norm gradient helper with
  # version 1 and returns its five values.
  return _BaseFusedBatchNormGrad(op, 1, *grad)


@ops.RegisterGradient("FusedBatchNormV3")
def _FusedBatchNormV3Grad(op: ops.Operation, *grad):
  # [FUNCTION DOC]
  # Gradient of FusedBatchNormV3. Forwards to the shared fused batch norm gradient helper with
  # version 2 (which also uses a third reserve space) and returns its five values.
  return _BaseFusedBatchNormGrad(op, 2, *grad)


@ops.RegisterGradient("L2Loss")
def _L2LossGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of the L2 loss. Inputs: op; grad, scalar gradient for the loss. Returns the op's
  # input multiplied by grad (broadcast), a tensor shaped like the input.
  """Return the gradients for L2Loss.

  Args:
    op: The L2LossOp for which we need to generate gradients.
    grad: Tensor containing a single number.

  Returns:
    The gradient, which is (x * grad).
  """
  return op.inputs[0] * grad


@ops.RegisterGradient("TopK")
@ops.RegisterGradient("TopKV2")
def _TopKGrad(op: ops.Operation, grad, _):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the second gradient is "all zero" with the shape of indices;
  # the code returns a scalar 32-bit integer zero for the k input, not a tensor of zeros.
  # Gradient of TopK and TopKV2 (registered for both). Inputs: op; grad, gradient for the values
  # output; the gradient for the indices output is ignored. Flattens the output indices to 2-D,
  # offsets each row by the row index times the last input dimension to get linear indices into
  # the flattened input, scatters the gradient values there into a zero tensor and reshapes to
  # the input shape. Returns a list: that tensor and a scalar zero. Indices are cast to 64-bit
  # for the offset arithmetic because 32-bit is unsupported on GPU.
  """Return the gradients for TopK.

  Args:
    op: The TopKOp for which we need to generate gradients.
    grad: Tensor. The gradients passed to the TopKOp.

  Returns:
    A list of two tensors, the first being the gradient w.r.t to the input and
    TopK, and the second being the gradient w.r.t. to the indices (all zero).
  """
  in_shape = array_ops.shape(op.inputs[0])
  ind_shape = array_ops.shape(op.outputs[1])

  # int32 is not supported on GPU hence up-casting
  ind_lastdim = array_ops.gather(
      math_ops.cast(ind_shape, dtypes.int64),
      array_ops.size(ind_shape) - 1)
  # Flatten indices to 2D.
  ind_2d = array_ops.reshape(
      op.outputs[1], array_ops_stack.stack([-1, ind_lastdim]))

  in_lastdim = array_ops.gather(
      math_ops.cast(in_shape, dtypes.int64),
      array_ops.size(in_shape) - 1)
  outerdim = array_ops.shape(ind_2d)[0]
  # Compute linear indices (flattened to 1D).
  ind = array_ops.reshape(
      ind_2d + math_ops.cast(
          array_ops.expand_dims(
              math_ops.range(0,
                             math_ops.cast(outerdim, dtypes.int64) * in_lastdim,
                             in_lastdim), -1), dtypes.int32), [-1])

  # Substitute grad to appropriate locations and fill the rest with zeros,
  # finally reshaping it to the original input shape.
  return [
      array_ops.reshape(
          array_ops.scatter_nd(
              array_ops.expand_dims(ind, -1), array_ops.reshape(grad, [-1]),
              [math_ops.reduce_prod(in_shape)]), in_shape),
      array_ops.zeros([], dtype=dtypes.int32)
  ]


@ops.RegisterGradient("ApproxTopK")
def _ApproxTopKGradient(op: ops.Operation, grad, _):
  # [FUNCTION DOC]
  # Gradient of approximate top-k. Inputs: op (output 1 holds indices; reduction-dimension
  # attribute read, negative values counted from the end); grad, gradient for the values output;
  # the indices gradient is ignored. Builds full coordinate tuples for each selected element
  # (the indices along the reduction dimension, iota ranges along the others), flattens them and
  # the gradient, and scatters into a tensor of the input's shape. Uses static shapes, so shapes
  # must be fully known. Returns that single tensor.
  """Return the gradients for ApproxTopK.

  Args:
    op: The ApproxTopK for which we need to generate gradients.
    grad: The gradients for backprop.

  Returns:
    Scattered gradient based on the top-k indices.
  """
  # The code below is to generate the correct index and value mapping for
  # scatter_nd to work properly.
  #
  # We use static evaluations as much as possible to reduce the runtime cost.
  # That's said, use operation.shape instead of array_ops.shape;
  # and use functools.reduce(operator.mul, ...) instead of math_ops.reduce_prod
  idx_shape = op.outputs[1].shape
  lifted_idx_shape = idx_shape + [1]
  flat_shape_len = functools.reduce(operator.mul, idx_shape)
  rank = idx_shape.rank
  reduction_dim = op.get_attr("reduction_dimension")
  if reduction_dim < 0:
    reduction_dim = rank + reduction_dim

  def GetLiftedIdx(d):
    # [FUNCTION DOC]
    # HELPER: builds one coordinate component for the approximate top-k gradient scatter.
    # Input: d, an axis number. For the reduction axis returns the op's index output with a
    # trailing axis added; for other axes returns an iota range along that axis, reshaped and
    # broadcast to the lifted index shape. Uses enclosing variables; no side effects.
    if d == reduction_dim:
      return array_ops.reshape(op.outputs[1], lifted_idx_shape)
    iota_len = idx_shape[d]
    iota_shape = list(itertools.repeat(1, rank + 1))
    iota_shape[d] = iota_len
    iota = array_ops.reshape(math_ops.range(iota_len), iota_shape)
    return array_ops.broadcast_to(iota, lifted_idx_shape)

  lifted_idx = array_ops.concat(
      list(GetLiftedIdx(d) for d in range(rank)), axis=rank)
  flat_idx = array_ops.reshape(lifted_idx, [flat_shape_len, rank])
  flat_grad = array_ops.reshape(grad, [flat_shape_len])
  return array_ops.scatter_nd(flat_idx, flat_grad, op.inputs[0].shape)


@ops.RegisterGradient("NthElement")
def _NthElementGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Gradient of NthElement. Inputs: op; grad, gradient for the selected element along the last
  # axis. Marks input entries equal to the selected value, divides the gradient evenly among ties
  # (the count of marked entries), and returns a list: the distributed gradient shaped like the
  # input, and nothing for the n input.
  """Return the gradients for NthElement.

  Args:
    op: The NthElementOp for which we need to generate gradients.
    grad: Tensor. The gradients passed to the NthElementOp

  Returns:
    A list of two tensors, the first being the gradient w.r.t. the input,
    the second being the gradient w.r.t. the N (None).
  """
  input = op.inputs[0]  # pylint: disable=redefined-builtin
  output = op.outputs[0]

  # Compute the number of elements which equal to output in each reduction
  # dimension. If there are multiple elements then the gradient will be
  # divided between them.
  indicators = math_ops.cast(
      math_ops.equal(array_ops.expand_dims(output, -1), input), grad.dtype)

  grad = array_ops.expand_dims(grad, -1)
  num_selected = array_ops.expand_dims(math_ops.reduce_sum(indicators, -1), -1)

  return [math_ops.divide(indicators, num_selected) * grad, None]


def _MeanAggregator(inputs, segments):
  # [FUNCTION DOC]
  # HELPER: used by the isotonic regression gradient.
  # KNOWN ISSUE: the docstring says segments has the same shape as "input" (a typo for inputs);
  # the code also requires the first dimension to be statically known, since it splits per row.
  # Inputs: inputs and segments, two 2-D tensors of equal shape (segments holds integer segment
  # ids). For each row computes the unsorted segment mean (avoiding segment-mean, unsupported on
  # TPU) and gathers it back so each element is replaced by its segment's mean. Returns the
  # stacked rows with the same shape and type as inputs.
  """Replaces each segment with its mean along the last axis.

  Specifically, each value in the `inputs` tensor gets replaced by the mean
  value computed from the values that belong to the same segment.

  Args:
   inputs: A 2-tensor. Aggregation is done over dimension 1.
   segments: A 2-tensor, same shape as `input`.

  Returns:
    The result, same shape and type as `inputs`.
  """
  result = []
  for inputs_i, segments_i in zip(
      array_ops.split(inputs, inputs.shape[0]),
      array_ops.split(segments, segments.shape[0])):
    # Note that we do not use tf.math.segment_mean, as it has no TPU support.
    means_i = math_ops.unsorted_segment_mean(
        inputs_i, segments_i, num_segments=math_ops.reduce_max(segments_i) + 1)
    result.append(
        array_ops.reshape(array_ops.gather(means_i, segments_i), [-1]))
  return array_ops_stack.stack(result, axis=0)


# We have to register the gradients for these ops so that tensorflow will know
# how to differentiate them.
@ops.RegisterGradient("IsotonicRegression")
def _IsotonicRegressionGrad(op: ops.Operation, grad_output, grad_segments):
  # [FUNCTION DOC]
  # Gradient of isotonic regression. Inputs: op (output 1 holds segment ids); grad_output,
  # gradient for the fitted values; grad_segments, gradient for the discrete segments, discarded.
  # Returns the segment-wise mean of grad_output (same shape as it) as the input gradient.
  """Gradient for the isotonic regression function.

  Args:
    op: The IsotonicRegression tensorflow op.
    grad_output: Tensor of incoming gradients with respect to the output.
    grad_segments: Tensor of incoming gradients with respect to the segments.

  Returns:
    A tensor, same size as `grad_output` with the gradient with respect to
    the input.
  """
  del grad_segments  # Discrete, non-differentiable.
  segments = op.outputs[1]
  return _MeanAggregator(grad_output, segments)
