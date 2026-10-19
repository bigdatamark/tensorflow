# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# Integration tests for TF-TensorRT conversion of the combined non-max-suppression op: the basic
# test, a variant executing the native (TensorFlow) segment, a top-k variant with many boxes and
# classes, and a variant allowing the top-k override via an environment variable. Several tests
# set and reset environment variables in setUp/tearDown. Built on TfTrtIntegrationTestBase.
"""Script to test TF-TensorRT conversion of CombinedNMS op."""

import os

from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.platform import test


class CombinedNmsTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that the combined non-max-suppression op (with constant threshold and size inputs) is
  # converted to an engine in static shape mode and not converted in dynamic shape mode. Holds the
  # number of boxes as an attribute. Subclasses vary the scenario.
  """Test for CombinedNMS op in TF-TRT."""

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup and sets the number of input boxes to 200.
    super().setUp()
    self.num_boxes = 200

  def GraphFn(self, boxes, scores):
    # [FUNCTION DOC]
    # Builds the graph from boxes and scores inputs: constants for max total size (3), IoU
    # threshold (0.5) and score threshold (0.1), the combined non-max-suppression op, and an
    # identity for each of its four outputs named output_0 to output_3. Returns the list of outputs.
    max_total_size = 3
    score_threshold = 0.1
    iou_threshold = 0.5
    # Shapes
    max_total_size_tensor = constant_op.constant(
        max_total_size, dtype=dtypes.int32, name='max_total_size')
    iou_threshold_tensor = constant_op.constant(
        iou_threshold, dtype=dtypes.float32, name='iou_threshold')
    score_threshold_tensor = constant_op.constant(
        score_threshold, dtype=dtypes.float32, name='score_threshold')
    nms_output = image_ops_impl.combined_non_max_suppression(
        boxes,
        scores,
        max_total_size_tensor,
        max_total_size_tensor,
        iou_threshold_tensor,
        score_threshold_tensor,
        name='combined_nms')
    return [
        array_ops.identity(output, name=('output_%d' % i))
        for i, output in enumerate(nms_output)
    ]

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, batch size 2, 2 classes, one box per class entry, num_boxes boxes;
    # inputs are boxes (batch, boxes, 1, 4) and scores (batch, boxes, classes); outputs are
    # boxes (2, 3, 4), scores (2, 3), classes (2, 3) and valid detection counts (2).
    # Parameters
    q = 1
    batch_size = 2
    num_classes = 2
    max_total_size = 3

    boxes_shape = [batch_size, self.num_boxes, q, 4]
    scores_shape = [batch_size, self.num_boxes, num_classes]
    nmsed_boxes_shape = [batch_size, max_total_size, 4]
    nmsed_scores_shape = [batch_size, max_total_size]
    nmsed_classes_shape = [batch_size, max_total_size]
    valid_detections_shape = [batch_size]
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [boxes_shape, scores_shape], [
                                nmsed_boxes_shape, nmsed_scores_shape,
                                nmsed_classes_shape, valid_detections_shape
                            ])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # In static shape mode expects one engine with the NMS op and its three constants; in dynamic
    # shape mode expects no engines (op not converted yet). Returns a mapping from engine name to
    # node name list (empty when none).
    """Return the expected engines to build."""
    if not run_params.dynamic_shape:
      return {
          'TRTEngineOp_000': [
              'combined_nms/CombinedNonMaxSuppression', 'max_total_size',
              'iou_threshold', 'score_threshold'
          ]
      }
    else:
      # The CombinedNMS op is currently not converted in dynamic shape mode.
      # This branch shall be removed once the converter is updated to handle
      # input with dynamic shape.
      return dict()

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Takes the parent decision and additionally requires a non-quantization precision and a linked
    # TensorRT version of at least 7.1.3. Returns a pair of boolean and reason string.
    should_run, reason = super().ShouldRunTest(run_params)
    should_run = should_run and \
        not trt_test.IsQuantizationMode(run_params.precision_mode)
    reason += ' and precision != INT8'
    # Only run for TRT 7.1.3 and above.
    return should_run and trt_utils.is_linked_tensorrt_version_greater_equal(
        7, 1, 3), reason + ' and >= TRT 7.1.3'


class CombinedNmsExecuteNativeSegmentTest(CombinedNmsTest):
  # [CLASS DOC]
  # Variant of the NMS test that allows native segment execution and builds the engine with a max
  # batch size smaller than the real one, forcing the runtime to run the TensorFlow implementation
  # of the NMS op inside the engine op (it has no GPU kernel). Static engines only.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then sets the environment variable allowing native segment execution
    # to true (process-wide).
    super().setUp()
    os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent teardown, then resets the native segment execution environment variable to
    # false.
    super().tearDown()
    os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'False'

  def GetMaxBatchSize(self, run_params):
    # [FUNCTION DOC]
    # Returns nothing for dynamic engines; otherwise the parent max batch size minus one so that
    # the actual batch exceeds the engine limit and the native segment runs.
    """Returns the max_batch_size that the converter should use for tests."""
    if run_params.dynamic_engine:
      return None

    # Build the engine with the allowed max_batch_size less than the actual
    # max_batch_size, to fore the runtime to execute the native segment. This
    # is to test that combined_non_max_suppression, which doesn't have a TF GPU
    # implementation, can be executed natively even though the it is in the
    # the graph for the TRTEngineOp with a GPU as a default device.
    return super().GetMaxBatchSize(run_params) - 1

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Takes the parent decision and additionally excludes dynamic engines. Returns a pair of
    # boolean and reason string.
    should_run, reason = super().ShouldRunTest(run_params)
    # max_batch_size is only useful for selecting static engines. As such,
    # we shouldn't run the test for dynamic engines.
    return should_run and \
        not run_params.dynamic_engine, reason + ' and static engines'


class CombinedNmsTestTopK(CombinedNmsTest):
  # [CLASS DOC]
  # NMS test with a large candidate count (2048 boxes, 90 classes) that exercises the top-k
  # selection path, with per-class padding disabled and the graph function taking the sizes as
  # extra arguments.
  """Test for CombinedNMS TopK op in TF-TRT."""

  def GraphFn(self, pre_nms_boxes, pre_nms_scores, max_boxes_to_draw,
              max_detetion_points):
    # [FUNCTION DOC]
    # Builds the graph: constants for max output size per class, max total size, IoU threshold
    # (0.1) and score threshold (0.001), the combined non-max-suppression op without per-class
    # padding, and an identity per output named output_0 to output_3. Inputs are the boxes and
    # scores tensors plus the integer sizes. Returns the list of outputs.

    iou_threshold = 0.1
    score_threshold = 0.001

    max_output_size_per_class_tensor = constant_op.constant(
        max_detetion_points,
        dtype=dtypes.int32,
        name='max_output_size_per_class')

    max_total_size_tensor = constant_op.constant(
        max_boxes_to_draw, dtype=dtypes.int32, name='max_total_size')

    iou_threshold_tensor = constant_op.constant(
        iou_threshold, dtype=dtypes.float32, name='iou_threshold')

    score_threshold_tensor = constant_op.constant(
        score_threshold, dtype=dtypes.float32, name='score_threshold')

    nms_output = image_ops_impl.combined_non_max_suppression(
        pre_nms_boxes,
        pre_nms_scores,
        max_output_size_per_class=max_output_size_per_class_tensor,
        max_total_size=max_total_size_tensor,
        iou_threshold=iou_threshold_tensor,
        score_threshold=score_threshold_tensor,
        pad_per_class=False,
        name='combined_nms')

    return [
        array_ops.identity(output, name=('output_%d' % i))
        for i, output in enumerate(nms_output)
    ]

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, batch size 1, 2048 candidate boxes, 90 classes, 30 boxes kept;
    # inputs are boxes (1, 2048, 1, 4) and scores (1, 2048, 90); outputs are boxes (1, 30, 4),
    # scores (1, 30), classes (1, 30) and valid counts (1). Wraps the graph function to fix the
    # size arguments.

    # Parameters
    batch_size = 1
    max_detetion_points = 2048
    num_classes = 90
    max_boxes_to_draw = 30

    # Inputs
    pre_nms_boxes_shape = [batch_size, max_detetion_points, 1, 4]
    pre_nms_scores_shape = [batch_size, max_detetion_points, num_classes]

    # Outputs
    nmsed_boxes_shape = [batch_size, max_boxes_to_draw, 4]
    nmsed_scores_shape = [batch_size, max_boxes_to_draw]
    nmsed_classes_shape = [batch_size, max_boxes_to_draw]
    valid_detections_shape = [batch_size]

    def _get_graph_fn(x, y):
      # [FUNCTION DOC]
      # HELPER: adapts the four-argument graph function to the two-input form expected by the
      # parameter builder, by fixing the max boxes and max detection point sizes. Returns the
      # outputs of the graph function.
      return self.GraphFn(
          x,
          y,
          max_boxes_to_draw=max_boxes_to_draw,
          max_detetion_points=max_detetion_points)

    return self.BuildParams(_get_graph_fn, dtypes.float32,
                            [pre_nms_boxes_shape, pre_nms_scores_shape], [
                                nmsed_boxes_shape, nmsed_scores_shape,
                                nmsed_classes_shape, valid_detections_shape
                            ])


class CombinedNmsTopKOverride(CombinedNmsTest):
  # [CLASS DOC]
  # Variant of the NMS test with 5000 boxes and the environment variable that allows overriding the
  # top-k limit enabled for the duration of each test.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, sets the number of boxes to 5000 and sets the environment variable
    # allowing the NMS top-k override to 1 (process-wide).
    super().setUp()
    self.num_boxes = 5000
    os.environ['TF_TRT_ALLOW_NMS_TOPK_OVERRIDE'] = '1'

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent teardown, then resets the top-k override environment variable to 0.
    super().tearDown()
    os.environ['TF_TRT_ALLOW_NMS_TOPK_OVERRIDE'] = '0'

  def GetMaxBatchSize(self, run_params):
    # [FUNCTION DOC]
    # Returns nothing for dynamic engines; otherwise the parent max batch size.
    """Returns the max_batch_size that the converter should use for tests."""
    if run_params.dynamic_engine:
      return None
    return super().GetMaxBatchSize(run_params)


if __name__ == '__main__':
  test.main()
