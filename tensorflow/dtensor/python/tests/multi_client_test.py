# [MODULE DOC]
# Multi-client DTensor tests. Each client process runs the same tests on a 2D mesh (batch by
# model, with the model dimension size set by a command-line flag defaulting to four) built from
# all clients' devices. Contains a tiny MNIST-style training comparison between plain variables and
# DTensor variables (model-sharded weight and bias), a host/device copy test, and a configuration
# function that the multi-client launcher calls in every client to set environment variables,
# create logical devices and initialize the accelerator system with the coordination service.
# Import-time side effect: defines the model dimension size flag. Running as a script goes through the
# multi-client test entry point of the test backend utilities.
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
"""Tests for DTensor multi-client setup."""
import os

from absl import flags
import numpy as np

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as d_layout
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import multi_client_test_util
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test


_MODEL_DIM_SIZE = flags.DEFINE_integer('model_dim_size', 4,
                                       'Size of the model dimension.')

_BATCH_DIM = 'batch'
_MODEL_DIM = 'model'

_BATCH_SIZE = 8
_STEPS = 5
_LR = 1e-3


@polymorphic_function.function
def _run_step(inputs, w, b, k):
  # [FUNCTION DOC]
  # HELPER: one training step (compiled as a traced function) shared by the plain and DTensor runs.
  # Inputs: image batch, dense weight variable w, bias variable b, convolution filter variable k.
  # Applies a same-padded convolution, flattens, multiplies by w, adds b, and sums to a scalar loss.
  # Computes gradients with respect to w and b only, subtracts the learning rate times each gradient
  # from the variable in place, and returns the weight gradient, bias gradient and loss.
  # Side effect: updates w and b.
  with backprop.GradientTape() as g:
    g.watch([w, b])
    logits = nn_ops.conv2d_v2(inputs, k, strides=[1, 1, 1, 1], padding='SAME')
    logits = array_ops.reshape(logits, [logits.shape[0], -1])
    logits = math_ops.matmul(logits, w)
    logits = logits + b
    loss = math_ops.reduce_sum(logits, axis=[0, 1])

  gw, gb = g.gradient(loss, [w, b])
  for v, v_grad in zip([w, b], [gw, gb]):
    v.assign_sub(_LR * v_grad)
  return gw, gb, loss


def init_var(mesh):
  # [FUNCTION DOC]
  # HELPER: creates matching plain and DTensor model variables for the MNIST test.
  # Input: the 2D mesh. Generates deterministic random normal initial values for a weight
  # (784 by 16), bias (16) and 3x3 one-channel filter. Returns a pair: ordinary variables
  # (weight, bias, filter) and DTensor variables (weight, bias, filter). The DTensor weight is
  # sharded on the model mesh dimension along its second axis, the bias is sharded on the model
  # dimension, and the filter is fully replicated.
  w_initializer = stateless_random_ops.stateless_random_normal(
      [28 * 28, 16], seed=[0, 1]
  )
  b_initializer = stateless_random_ops.stateless_random_normal(
      [16], seed=[0, 2]
  )
  k_initializer = stateless_random_ops.stateless_random_normal(
      [3, 3, 1, 1], seed=[0, 3]
  )
  n_w = variables.Variable(w_initializer)
  n_b = variables.Variable(b_initializer)
  n_k = variables.Variable(k_initializer)

  # Initialize DTensor variables.
  w_initializer_on_mesh = d_api.copy_to_mesh(
      w_initializer, d_layout.Layout.replicated(mesh, rank=2)
  )
  b_initializer_on_mesh = d_api.copy_to_mesh(
      b_initializer, d_layout.Layout.replicated(mesh, rank=1)
  )
  k_initializer_on_mesh = d_api.copy_to_mesh(
      k_initializer, d_layout.Layout.replicated(mesh, rank=4)
  )
  w = d_variable.DVariable(
      d_api.relayout(
          w_initializer_on_mesh,
          d_layout.Layout(['unsharded', _MODEL_DIM], mesh),
      )
  )
  b = d_variable.DVariable(
      d_api.relayout(b_initializer_on_mesh, d_layout.Layout([_MODEL_DIM], mesh))
  )
  k = d_variable.DVariable(k_initializer_on_mesh)
  return (n_w, n_b, n_k), (w, b, k)


class DTensorMNISTTest(tf_test.TestCase):
  # [CLASS DOC]
  # Multi-client test case run identically in every client process. Holds a 2D mesh (batch by
  # model) spanning all clients' devices, stored as the mesh attribute during setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: calls the base setUp, picks the device type (GPU if physical GPUs exist, TPU
    # if present, otherwise CPU), lists local devices, builds a global device id grid whose rows are
    # clients times local devices divided by the model dimension size flag and whose columns are the
    # model dimension size, takes this client's slice of the ids, and creates the mesh with batch
    # and model dimensions stored on the instance.
    super(DTensorMNISTTest, self).setUp()
    if config.list_physical_devices('GPU'):
      device_type = 'GPU'
    elif test_util.is_tpu_present():
      device_type = 'TPU'
    else:
      device_type = 'CPU'

    local_devices = d_config.local_devices(device_type)
    num_devices = len(local_devices)

    global_device_ids = test_util.create_device_ids_array((
        d_config.num_clients() * num_devices // _MODEL_DIM_SIZE.value,
        _MODEL_DIM_SIZE.value,
    ))
    device_ids_list = np.ravel(global_device_ids).tolist()

    index = d_config.client_id() * num_devices
    local_device_ids = device_ids_list[index:(index + num_devices)]

    self.mesh = d_layout.Mesh(
        [_BATCH_DIM, _MODEL_DIM],
        global_device_ids=global_device_ids,
        local_device_ids=local_device_ids,
        local_devices=local_devices,
    )

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: runs a barrier across the mesh so no client disconnects before the others
    # finish, then calls the base tearDown.
    # A barrier prevents a client from disconnecting prematurely in tests.
    mesh_util.barrier(self.mesh)
    super().tearDown()

  def test_mnist(self):
    # [FUNCTION DOC]
    # Test: trains the small convolution plus dense model for five steps with ordinary variables
    # and with DTensor variables (batch-sharded random inputs) via a nested training function, then
    # asserts the final loss, weight gradient, bias gradient, weight and bias of the DTensor run
    # match the plain run within small absolute tolerances.
    def train():
      # [FUNCTION DOC]
      # Nested training routine. For five steps, draws a deterministic random batch of eight 28x28x1
      # images, runs a step on the plain variables, copies the batch to the mesh, relays it
      # out batch-sharded and runs a step on the DTensor variables on the first local device. After
      # the loop relays the final DTensor gradients and variables to replicated layouts. Returns a
      # pair of tuples: (plain loss, weight gradient, bias gradient, weight, bias) and the
      # same five values for the DTensor run.
      input_layout = d_layout.Layout.batch_sharded(
          self.mesh, _BATCH_DIM, rank=4
      )
      (n_w, n_b, n_k), (w, b, k) = init_var(self.mesh)
      for i in range(_STEPS):
        data = stateless_random_ops.stateless_random_normal(
            [_BATCH_SIZE, 28, 28, 1], seed=[0, i]
        )
        g_nw, g_nb, n_loss = _run_step(data.numpy(), n_w, n_b, n_k)

        input_image = d_api.copy_to_mesh(
            data, layout=d_layout.Layout.replicated(self.mesh, rank=4)
        )
        input_image = d_api.relayout(input_image, layout=input_layout)

        with ops.device_v2(self.mesh.local_devices()[0]):
          gw, gb, loss = _run_step(input_image, w, b, k)

      gw = d_api.relayout(gw, d_layout.Layout.replicated(self.mesh, rank=2))
      w = d_api.relayout(w, d_layout.Layout.replicated(self.mesh, rank=2))
      gb = d_api.relayout(gb, d_layout.Layout.replicated(self.mesh, rank=1))
      b = d_api.relayout(b, d_layout.Layout.replicated(self.mesh, rank=1))

      return (n_loss, g_nw, g_nb, n_w, n_b), (loss, gw, gb, w, b)

    (n_loss, g_nw, g_nb, n_w, n_b), (loss, gw, gb, w, b) = train()

    self.assertAllClose(n_loss, loss, atol=5e-4)
    self.assertAllClose(g_nw, gw, atol=1e-5)
    self.assertAllClose(g_nb, gb, atol=1e-5)
    self.assertAllClose(n_w, w, atol=1e-5)
    self.assertAllClose(n_b, b, atol=1e-5)

  def test_copy_to_mesh(self):
    # [FUNCTION DOC]
    # Test: ones of shape 2x2 are packed onto the mesh with a batch-sharded layout, copied to the
    # host mesh with a traced function, unpacked and each component checked to equal ones; then
    # copied back to the device mesh with another traced function and checked again.
    layout = d_layout.Layout([_BATCH_DIM, 'unsharded'], self.mesh)
    host_layout = d_layout.Layout(layout.sharding_specs, self.mesh.host_mesh())
    x = d_api.pack(
        [array_ops.ones((2, 2), dtype=dtypes.float32)]
        * len(self.mesh.local_devices()),
        layout,
    )

    @polymorphic_function.function
    def d2h(x):
      # [FUNCTION DOC]
      # Traced function: copies the input DTensor to the host mesh with the host layout.
      return d_api.copy_to_mesh(x, host_layout)

    @polymorphic_function.function
    def h2d(x):
      # [FUNCTION DOC]
      # Traced function: copies the input DTensor from the host mesh back to the device mesh layout.
      return d_api.copy_to_mesh(x, layout)

    y = d2h(x)
    ys = d_api.unpack(y)
    for i in ys:
      self.assertAllClose(i, array_ops.ones((2, 2)), atol=1e-5)
    z = h2d(y)
    zs = d_api.unpack(z)
    for i in zs:
      self.assertAllClose(i, array_ops.ones((2, 2)), atol=1e-5)


def client_config_function(config_params):
  # [FUNCTION DOC]
  # HELPER: per-client configuration callback passed to the multi-client test launcher.
  # Input: a mapping with keys for the number of clients, this client's id, the list of worker
  # job addresses and the number of devices per client. When the client count is nonzero, sets the
  # DTensor client id, job name (worker) and job list environment variables. Chooses the device
  # type (GPU, then TPU, else CPU), resets the TensorFlow context, creates virtual logical devices
  # for non-TPU types, and initializes the accelerator system with the coordination service
  # enabled. Finally asserts (assertion error) that the number of local logical devices equals the
  # requested number. Returns nothing; side effects are environment variables and global state.
  num_clients = config_params['num_clients']
  dtensor_client_id = config_params['client_id']
  dtensor_jobs = config_params['worker_jobs']
  num_devices = config_params['num_devices']

  # pylint: disable=protected-access
  if num_clients != 0:
    os.environ[d_config._DT_CLIENT_ID] = f'{dtensor_client_id}'
    os.environ[d_config._DT_JOB_NAME] = 'worker'
    os.environ[d_config._DT_JOBS] = ','.join(dtensor_jobs)
  # pylint: enable=protected-access

  if config.list_physical_devices('GPU'):
    device_type = 'GPU'
  elif test_util.is_tpu_present():
    device_type = 'TPU'
  else:
    device_type = 'CPU'

  # reset_logical_devices
  test_util.reset_context()
  if device_type != 'TPU':
    # Configure virtual devices. This does not initialize the TensorFlow
    # context.
    test_util.reset_logical_devices(device_type, num_devices)
  accelerator_util.initialize_accelerator_system(
      device_type, enable_coordination_service=True)

  # Validates the correct number of devices are created.
  logical_devices = test_util.list_local_logical_devices(device_type)
  assert len(logical_devices) == num_devices, (
      logical_devices,
      f'Test is mis-configured: expecting {num_devices} logical_devices.')


if __name__ == '__main__':
  test_backend_util.handle_test_main(
      multi_client_test_util.multi_client_main, client_config_function)
