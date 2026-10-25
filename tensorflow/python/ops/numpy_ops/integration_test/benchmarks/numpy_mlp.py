# [MODULE DOC]
# HELPER: benchmark support module. Defines the default sizes (input size 10, hidden units 10,
# 3 classes) and a small two-layer perceptron implemented with plain numpy, used as the numpy
# baseline in the micro benchmarks. No import-time side effects beyond constants.
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
"""Builds the MLP network."""
import numpy as np

NUM_CLASSES = 3
INPUT_SIZE = 10
HIDDEN_UNITS = 10


class MLP:
  # [CLASS DOC]
  # HELPER: benchmark model. Two-layer perceptron with randomly initialized (uniform) float32
  # weights w1, w2 and biases b1, b2 held as numpy arrays. Created with optional sizes and used
  # through inference.
  """MLP model.

  T = Relu(Add(MatMul(A, B), C))
  R = Relu(Add(MatMul(T, D), E))
  """

  def __init__(self, num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
               hidden_units=HIDDEN_UNITS):
    # [FUNCTION DOC]
    # Creates the parameters: inputs are the number of output classes, the input width and the
    # hidden width (defaults 3, 10, 10). Draws uniform random float32 arrays for the first-layer
    # weights (input by hidden), second-layer weights (hidden by classes) and row-vector biases.
    # Uses the global numpy random state. Returns nothing.
    self.w1 = np.random.uniform(size=[input_size, hidden_units]).astype(
        np.float32, copy=False)
    self.w2 = np.random.uniform(size=[hidden_units, num_classes]).astype(
        np.float32, copy=False)
    self.b1 = np.random.uniform(size=[1, hidden_units]).astype(
        np.float32, copy=False)
    self.b2 = np.random.uniform(size=[1, num_classes]).astype(
        np.float32, copy=False)

  def inference(self, inputs):
    # [FUNCTION DOC]
    # Runs the forward pass on a batch of inputs (rows of width input size) using the stored
    # parameters; returns the batch of outputs with one column per class.
    return self._forward(inputs, self.w1, self.w2, self.b1, self.b2)

  def _forward(self, x, w1, w2, b1, b2):
    # [FUNCTION DOC]
    # HELPER: forward computation for inference.
    # Applies two layers, each a matrix product plus bias followed by clipping at zero (relu).
    # Takes the input batch and the two weight matrices and two biases; returns the output array.
    # Pure computation.
    x = np.maximum(np.matmul(x, w1) + b1, 0.)
    x = np.maximum(np.matmul(x, w2) + b2, 0.)
    return x
