# [MODULE DOC]
# KNOWN ISSUE: IMPLEMENTED_OPS omits BatchMatMul, BatchMatMulV2 and BatchMatMulV3 although
# flops functions are registered for them below, so the set is incomplete.
# Registers floating-point-operation (flops) statistics functions for individual TensorFlow op
# types with the global statistics registry (ops.RegisterStatistics, statistic type "flops").
# Each registered function takes a graph and a node definition and returns an OpStats record
# named "flops". Shapes are looked up through graph_util and must be fully defined, otherwise
# the shape assertion fails. Covers unary, binary, reduction, pooling, convolution-gradient,
# AddN, MatMul and BatchMatMul ops; convolution/pooling support is limited to NHWC layout.
# Exposes IMPLEMENTED_OPS, a set of op names with flops support. Importing the module registers
# all statistics as a side effect. Used by the profiler's model analysis code.
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
"""Register flops statistics for various TensorFlow operations.
"""
import numpy as np

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops


# List of all ops which have implemented flops statistics.
IMPLEMENTED_OPS = set([
    # Unary ops
    "Reciprocal", "Square", "Rsqrt", "Log", "Neg", "AssignSub", "AssignAdd",
    "L2Loss", "Softmax",
    # Binary ops
    "Add", "Sub", "Mul", "RealDiv", "Maximum", "Minimum", "Pow", "RsqrtGrad",
    "GreaterEqual", "Greater", "LessEqual", "Less", "Equal", "NotEqual",
    "SquaredDifference", "AddV2",
    # Reduction ops
    "Mean", "Sum", "ArgMax", "ArgMin", "BiasAddGrad",
    # Convolution and pooling
    "AvgPool", "MaxPool", "AvgPoolGrad", "MaxPoolGrad", "Conv2DBackpropInput",
    "Conv2DBackpropFilter",
    # Other ops
    "AddN", "MatMul",
    # Ops implemented in core tensorflow:
    "Conv2D", "DepthwiseConv2dNative", "BiasAdd", "Dilation2D",
])


def _zero_flops(graph, node):
  # [FUNCTION DOC]
  # HELPER: fallback used by flops statistics when an op has nothing to count.
  # Ignores both arguments and returns an OpStats record named flops with value zero.
  """Returns zero flops."""
  del graph, node  # graph and node are unused
  return ops.OpStats("flops", 0)


def _list_product(lst):
  # [FUNCTION DOC]
  # HELPER: arithmetic utility for the pooling flops functions.
  # Takes a list of numbers and returns their product; an empty list yields one. No side effects.
  """Computes product of element of the list."""
  result = 1
  for item in lst:
    result *= item
  return result

################################################################################
# Unary operations
################################################################################


def _unary_op_flops(graph, node, ops_per_element=1):
  # [FUNCTION DOC]
  # HELPER: shared implementation for all element-wise unary op flops functions.
  # Looks up the shape of the node's first input in the graph (must be fully defined, otherwise
  # the shape assertion raises) and returns an OpStats record named flops equal to the input
  # element count times ops_per_element (default 1, number of flops spent per element).
  """Common code which compute flops for unary operations."""
  in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  in_shape.assert_is_fully_defined()
  return ops.OpStats("flops", in_shape.num_elements() * ops_per_element)


@ops.RegisterStatistics("Reciprocal", "flops")
def _reciprocal_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Reciprocal: one flop per input element, via the unary helper.
  # Returns an OpStats record; raises if the input shape is not fully defined.
  """Compute flops for Reciprocal operation."""
  return _unary_op_flops(graph, node)


@ops.RegisterStatistics("Square", "flops")
def _square_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Square: one flop per input element, via the unary helper.
  # Returns an OpStats record; raises if the input shape is not fully defined.
  """Compute flops for Square operation."""
  return _unary_op_flops(graph, node)


@ops.RegisterStatistics("Rsqrt", "flops")
def _rsqrt_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Rsqrt: counted as a square root plus a reciprocal, so two
  # flops per input element. Returns an OpStats record; raises if the input shape is unknown.
  """Compute flops for Rsqrt operation."""
  # Rsqrt(x) = 1 / sqrt(x)
  return _unary_op_flops(graph, node, ops_per_element=2)


@ops.RegisterStatistics("Log", "flops")
def _log_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Log: one flop per input element, via the unary helper.
  # Returns an OpStats record; raises if the input shape is not fully defined.
  """Compute flops for Log operation."""
  return _unary_op_flops(graph, node)


@ops.RegisterStatistics("Neg", "flops")
def _neg_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Neg: one flop per input element, via the unary helper.
  # Returns an OpStats record; raises if the input shape is not fully defined.
  """Compute flops for Neg operation."""
  return _unary_op_flops(graph, node)


@ops.RegisterStatistics("AssignSub", "flops")
def _assign_sub_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for AssignSub: one flop per element of the first input.
  # Returns an OpStats record; raises if the input shape is not fully defined.
  """Compute flops for AssignSub operation."""
  return _unary_op_flops(graph, node)


@ops.RegisterStatistics("AssignAdd", "flops")
def _assign_add_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for AssignAdd: one flop per element of the first input.
  # Returns an OpStats record; raises if the input shape is not fully defined.
  """Compute flops for AssignAdd operation."""
  return _unary_op_flops(graph, node)


@ops.RegisterStatistics("L2Loss", "flops")
def _l2_loss_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for L2Loss. Reads the first input shape (must be fully defined)
  # and returns an OpStats record with three times the element count minus one flops, matching
  # the less efficient implementation used by TensorFlow rather than the optimal two per element.
  """Compute flops for L2Loss operation."""
  in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  in_shape.assert_is_fully_defined()
  # Tensorflow uses inefficient implementation, with (3*N-1) flops:
  # Optimal implementation is 2*N flops
  return ops.OpStats("flops", in_shape.num_elements() * 3 - 1)


@ops.RegisterStatistics("Softmax", "flops")
def _softmax_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Softmax: five flops per input element (two for shifting the
  # logits, one for the exponential, two for normalizing). Returns an OpStats record; raises if
  # the input shape is not fully defined.
  """Compute flops for Softmax operation."""
  # Softmax implementation:
  #
  # Approximate flops breakdown:
  #   2*n          -- compute shifted logits
  #   n            -- exp of shifted logits
  #   2*n          -- compute softmax from exp of shifted logits
  return _unary_op_flops(graph, node, ops_per_element=5)

################################################################################
# Binary operations
################################################################################


def _binary_per_element_op_flops(graph, node, ops_per_element=1):
  # [FUNCTION DOC]
  # HELPER: shared implementation for element-wise binary op flops functions.
  # Uses the node's output shape (must be fully defined, otherwise the assertion raises) and
  # returns an OpStats record named flops equal to output element count times ops_per_element
  # (default 1).
  """Common code which compute flops for binary operations."""
  out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  out_shape.assert_is_fully_defined()
  return ops.OpStats("flops", out_shape.num_elements() * ops_per_element)


@ops.RegisterStatistics("Add", "flops")
@ops.RegisterStatistics("AddV2", "flops")
def _add_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for both Add and AddV2: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Add operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Sub", "flops")
def _sub_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Sub: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Sub operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Mul", "flops")
def _mul_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Mul: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Mul operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("RealDiv", "flops")
def _real_div_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for RealDiv: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for RealDiv operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Maximum", "flops")
def _maximum_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Maximum: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Maximum operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Minimum", "flops")
def _minimum_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Minimum: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Minimum operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Pow", "flops")
def _pow_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Pow: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Pow operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("RsqrtGrad", "flops")
def _rsqrt_grad_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for RsqrtGrad: four flops per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for RsqrtGrad operation."""
  return _binary_per_element_op_flops(graph, node, ops_per_element=4)


@ops.RegisterStatistics("GreaterEqual", "flops")
def _greater_equal_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for GreaterEqual: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for GreaterEqual operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Greater", "flops")
def _greater_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Greater: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Greater operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("LessEqual", "flops")
def _less_equal_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for LessEqual: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for LessEqual operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Less", "flops")
def _less_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Less: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Less operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("Equal", "flops")
def _equal_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Equal: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for Equal operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("NotEqual", "flops")
def _not_equal_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for NotEqual: one flop per output element.
  # Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for NotEqual operation."""
  return _binary_per_element_op_flops(graph, node)


@ops.RegisterStatistics("SquaredDifference", "flops")
def _squared_difference_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for SquaredDifference: two flops (subtract, square) per output
  # element. Returns an OpStats record; raises if the output shape is not fully defined.
  """Compute flops for SquaredDifference operation."""
  return _binary_per_element_op_flops(graph, node, ops_per_element=2)

################################################################################
# Reduction ops
################################################################################


def _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0):
  # [FUNCTION DOC]
  # HELPER: shared implementation for reduction op flops functions.
  # Needs fully defined shapes for the first input and the output (assertion raises otherwise).
  # Flops are input elements times reduce_flops (default 1) plus output elements times the
  # difference finalize_flops (default 0) minus reduce_flops, i.e. the first element of each
  # reduced group costs no reduction step. Returns an OpStats record named flops.
  """Common code which compute flops for reduction operations."""
  in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  in_shape.assert_is_fully_defined()
  out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  out_shape.assert_is_fully_defined()
  num_flops = (in_shape.num_elements() * reduce_flops
               + out_shape.num_elements() * (finalize_flops - reduce_flops))
  return ops.OpStats("flops", num_flops)


@ops.RegisterStatistics("Mean", "flops")
def _mean_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Mean: reduction by summation (one flop) with a one flop
  # division finalization per output element. Returns an OpStats record; shapes must be known.
  """Compute flops for Mean operation."""
  # reduction - sum, finalization - divide
  return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=1)


@ops.RegisterStatistics("Sum", "flops")
def _sum_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Sum: one flop per reduction step and no finalization.
  # Returns an OpStats record; shapes must be fully defined.
  """Compute flops for Sum operation."""
  # reduction - sum, no finalization
  return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)


@ops.RegisterStatistics("ArgMax", "flops")
def _arg_max_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for ArgMax: one comparison per reduction step, no finalization.
  # Returns an OpStats record; shapes must be fully defined.
  """Compute flops for ArgMax operation."""
  # reduction - comparison, no finalization
  return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)


@ops.RegisterStatistics("ArgMin", "flops")
def _arg_min_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for ArgMin: one comparison per reduction step, no finalization.
  # Returns an OpStats record; shapes must be fully defined.
  """Compute flops for ArgMin operation."""
  # reduction - comparison, no finalization
  return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)


@ops.RegisterStatistics("BiasAddGrad", "flops")
def _bias_add_grad_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for BiasAddGrad, which is essentially a reduce-sum, so it is
  # costed exactly like Sum. Returns an OpStats record; shapes must be fully defined.
  """Compute flops for BiasAddGrad operation."""
  # Implementation of BiasAddGrad, essentially it's a reduce sum and reshaping:
  # So computing flops same way as for "Sum"
  return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)

################################################################################
# Convolution and pooling
# Note: all flops statistics are implemented only for NHWC data format
################################################################################


def _verify_conv_data_format(node):
  # [FUNCTION DOC]
  # HELPER: guard used by the pooling and convolution flops functions.
  # Checks that the node's data format attribute is the NHWC layout and raises a value error
  # saying only NHWC is supported otherwise. Returns nothing.
  """Verifies data format for pooling and convolutional operations."""
  # TODO(xpan): P1: Support NCHW
  if node.attr["data_format"].s != b"NHWC":
    raise ValueError("Only NHWC format is supported in flops computations")


def _pool_flops(graph, node):
  # [FUNCTION DOC]
  # HELPER: shared implementation for AvgPool and MaxPool flops.
  # Verifies NHWC format (value error otherwise), requires a fully defined output shape, and
  # returns an OpStats record equal to the product of the kernel size entries times the number
  # of output elements.
  """Common code which compute flops for pooling operations."""
  # compute flops for average and max pooling
  _verify_conv_data_format(node)
  #
  # Pooling declaration:
  #   Inputs:
  #     - value
  #   Outputs:
  #     - output
  #   Attributes:
  #     - ksize
  #     - strides
  #     - padding
  #     - data_format
  #
  # Pooling implementation:
  out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  out_shape.assert_is_fully_defined()
  kernel_shape = list(node.attr["ksize"].list.i)
  kernel_area = _list_product(kernel_shape)
  return ops.OpStats("flops", kernel_area * out_shape.num_elements())


@ops.RegisterStatistics("AvgPool", "flops")
def _avg_pool_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for AvgPool; delegates to the common pooling computation.
  # Raises a value error for non-NHWC data or if the output shape is not fully defined.
  """Compute flops for AvgPool operation."""
  return _pool_flops(graph, node)


@ops.RegisterStatistics("MaxPool", "flops")
def _max_pool_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for MaxPool; delegates to the common pooling computation.
  # Raises a value error for non-NHWC data or if the output shape is not fully defined.
  """Compute flops for MaxPool operation."""
  return _pool_flops(graph, node)


@ops.RegisterStatistics("AvgPoolGrad", "flops")
def _avg_pool_grad_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for AvgPoolGrad. Verifies NHWC format, requires the second
  # input (output backprop) shape to be fully defined, and returns an OpStats record equal to
  # kernel area times backprop element count times two (a multiply and an add per element).
  """Compute flops for AvgPoolGrad operation."""
  _verify_conv_data_format(node)
  # Pooling gradient implementation:
  out_backprop_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                                  node.input[1])
  out_backprop_shape.assert_is_fully_defined()
  kernel_shape = list(node.attr["ksize"].list.i)
  kernel_area = _list_product(kernel_shape)
  # TensorFlow multiply each element of pooling window by coefficient,
  # then sum up all of them, thus we have 2 flops per element:
  # More optimal implementation - if division is done after.
  return ops.OpStats("flops",
                     kernel_area * out_backprop_shape.num_elements() * 2)


@ops.RegisterStatistics("MaxPoolGrad", "flops")
def _max_pool_grad_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for MaxPoolGrad. Verifies NHWC format, requires the original
  # output (second input) shape to be fully defined, and returns an OpStats record equal to the
  # cost of recomputing the max pool (kernel area times those elements) plus one flop for each
  # element of the original output.
  """Compute flops for MaxPoolGrad operation."""
  _verify_conv_data_format(node)
  #
  # MaxPoolGrad declaration:
  #   Inputs:
  #     - orig_input  -- original input tensor (of max_pool)
  #     - orig_output  -- original output tensor (of max_pool)
  #     - grad --  gradient with respect to output of max_pool
  #   Outputs:
  #     - output -- gradient with respect to input of max_pool
  #   Attributes:
  #     - ksize
  #     - strides
  #     - padding
  #     - data_format
  # It computes MaxPool first, then one flop per each element of original output
  #
  kernel_shape = list(node.attr["ksize"].list.i)
  kernel_area = _list_product(kernel_shape)
  orig_out_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                              node.input[1])
  orig_out_shape.assert_is_fully_defined()
  max_pool_ops = kernel_area * orig_out_shape.num_elements()
  return ops.OpStats("flops", max_pool_ops + orig_out_shape.num_elements())


@ops.RegisterStatistics("Conv2DBackpropInput", "flops")
def _conv_2d_backprop_input_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Conv2DBackpropInput. Verifies NHWC format; needs the output
  # shape and the filter shape (second input) to be fully defined. Returns an OpStats record of
  # two times output elements times filter elements, divided by the output depth and the
  # product of the height and width strides. The result may be a floating point value.
  """Compute flops for Conv2DBackpropInput operation."""
  # Formula:
  #  batch_size * image_x_dim * image_y_dim * kernel_x_dim * kernel_y_dim
  #  * input_depth * output_depth * 2 / (image_x_stride * image_x_stride)
  #
  # Where:
  # image_x_dim, image_y_dim and input_depth --- size of input to source (no
  #   backprop) convolution, in other words they are sizes of backprop output.
  # output_depth --- number of filters in the original convolution, thus
  #   depth of backprop input.
  # kernel_x_dim and kernel_y_dim --- sizes of filter in spatial dimension
  # image_x_stride and image_x_stride --- strides of the convolution
  #
  _verify_conv_data_format(node)
  # out_shape = [batch_size, image_y_dim, image_x_dim, input_depth]
  out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  out_shape.assert_is_fully_defined()
  # kernel_shape = [kernel_y_dim, kernel_x_dim, input_depth, output_depth]
  kernel_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                            node.input[1])
  kernel_shape.assert_is_fully_defined()
  # strides
  strides_shape = list(node.attr["strides"].list.i)
  strides_product = strides_shape[1] * strides_shape[2]
  return ops.OpStats("flops",
                     (2 * out_shape.num_elements()
                      * kernel_shape.num_elements()
                      / (out_shape.dims[-1].value * strides_product)))


@ops.RegisterStatistics("Conv2DBackpropFilter", "flops")
def _conv_2d_backprop_filter_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for Conv2DBackpropFilter. Verifies NHWC format; needs the image
  # (first input) shape and the filter (node output) shape to be fully defined. Returns an
  # OpStats record of two times image elements times filter elements, divided by the image depth
  # and the product of the height and width strides. The result may be a floating point value.
  """Compute flops for Conv2DBackpropFilter operation."""
  # Formula same as for Conv2DBackpropInput:
  #  batch_size * image_x_dim * image_y_dim * kernel_x_dim * kernel_y_dim
  #  * input_depth * output_depth * 2 / (image_x_stride * image_x_stride)
  #
  _verify_conv_data_format(node)
  # image_shape = [batch_size, image_y_dim, image_x_dim, input_depth]
  image_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  image_shape.assert_is_fully_defined()
  # kernel_shape = [kernel_y_dim, kernel_x_dim, input_depth, output_depth]
  kernel_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  kernel_shape.assert_is_fully_defined()
  # strides
  strides_shape = list(node.attr["strides"].list.i)
  strides_product = strides_shape[1] * strides_shape[2]
  return ops.OpStats("flops",
                     (2 * image_shape.num_elements()
                      * kernel_shape.num_elements()
                      / (image_shape.dims[-1].value * strides_product)))

################################################################################
# Other ops
################################################################################


@ops.RegisterStatistics("AddN", "flops")
def _add_n_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for AddN. With no inputs returns zero flops; otherwise requires
  # the first input shape to be fully defined and returns element count times one less than the
  # number of inputs.
  """Compute flops for AddN operation."""
  if not node.input:
    return _zero_flops(graph, node)
  in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  in_shape.assert_is_fully_defined()
  return ops.OpStats("flops", in_shape.num_elements() * (len(node.input) - 1))


@ops.RegisterStatistics("MatMul", "flops")
def _calc_mat_mul_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for MatMul. Reads the transpose_a attribute to pick the inner
  # dimension of the first input (first dimension if transposed, otherwise second), requires the
  # input and output shapes be fully defined, and returns an OpStats record equal to the inner
  # dimension times the output element count times two (multiply and add).
  """Calculates the compute resources needed for MatMul."""
  transpose_a = node.attr["transpose_a"].b
  a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  a_shape.assert_is_fully_defined()
  if transpose_a:
    k = int(a_shape[0])
  else:
    k = int(a_shape[1])
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (k * output_count * 2))


@ops.RegisterStatistics("BatchMatMul", "flops")
@ops.RegisterStatistics("BatchMatMulV2", "flops")
@ops.RegisterStatistics("BatchMatMulV3", "flops")
def _calc_batch_mat_mul_flops(graph, node):
  # [FUNCTION DOC]
  # Registered flops statistic for BatchMatMul, BatchMatMulV2 and BatchMatMulV3. Same as the
  # plain matrix multiply cost but the inner dimension is taken from the last two dimensions of
  # the first input (second to last if transposed, else last). Shapes must be fully defined.
  # Returns an OpStats record: inner dimension times output elements times two.
  """Calculates the compute resources needed for BatchMatMul."""
  transpose_a = node.attr["transpose_a"].b
  a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  a_shape.assert_is_fully_defined()
  if transpose_a:
    k = int(a_shape[-2])
  else:
    k = int(a_shape[-1])
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (k * output_count * 2))
