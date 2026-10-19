# [MODULE DOC]
# HELPER: shared utilities for writing DTensor unit tests.
# Provides device-array and device-list builders, context and logical-device reset helpers,
# a base test class with mesh configuration, skip helpers and a DTensor equality assertion,
# a parameter-product helper and a DTensor singleton reset helper. Re-exports a few config
# predicates (TPU/GPU presence, preferred device type, multi-device mode) and the test backend
# name constants.
# Import-time side effects: enables TensorFlow V2 behavior globally and defines module
# constants (default tolerance, default GPU memory limit, flags handle).
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utility methods for DTensor testing."""

import collections
import copy
import itertools
import json
import os
import typing

from absl import flags
from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.config import is_gpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import is_tpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import preferred_device_type  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import use_multi_device_mode  # pylint: disable=unused-import
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.dtensor.python.tests.test_backend_name import DTensorTestUtilBackend
from tensorflow.dtensor.python.tests.test_backend_util import DTensorTestBackendConfigurator
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test as tf_test
from tensorflow.python.util import numpy_compat


# pylint: enable=g-direct-tensorflow-import

# DTensor only runs with TF V2.
v2_compat.enable_v2_behavior()

DEFAULT_TOL = 1e-5

_DEFAULT_GPU_MEMORY_LIMIT = 1024  # 1G


def get_use_xla_spmd(device_type):
  # [FUNCTION DOC]
  # Decides whether XLA SPMD should be used in tests. Input: device_type, a string naming the
  # device type of the mesh. Returns true only if the device type is exactly the upper-case TPU
  # name and the test XLA-SPMD environment variable is set to something other than zero (unset
  # counts as zero). Otherwise returns false. Reads the environment; no other side effects.
  """Returns True when device_type is TPU and environment variable is set.

  Args:
    device_type: A str representing the type of device on the mesh.

  Returns:
    bool: True when device_type is TPU and environment variable is set.
  """
  return device_type == 'TPU' and '0' != os.environ.get(
      'DTENSOR_TEST_USE_XLA_SPMD', '0'
  )


def create_device_ids_array(shape):
  # [FUNCTION DOC]
  # HELPER: builds device-id grids for mesh construction in tests.
  # Input: shape, a sequence of integers giving the mesh dimensions. Returns a numeric array of
  # that shape holding consecutive device ids starting at zero in row-major order.
  device_count = np.prod(shape)
  return np.arange(device_count).reshape(shape)


def create_device_array(shape, device_type):
  # [FUNCTION DOC]
  # HELPER: builds device-spec grids for mesh construction in tests.
  # Inputs: shape (sequence of integers, the mesh dimensions) and device_type (string such as
  # CPU, GPU or TPU). Returns an array of that shape whose entries are device specifications on
  # the local job, replica zero, task zero, with consecutive device indices in row-major order.
  device_count = np.prod(shape)
  return numpy_compat.np_asarray([
      tf_device.DeviceSpec(  # pylint: disable=g-complex-comprehension
          job='localhost/replica:0/task:0',
          device_type=device_type,
          device_index=i,
      )
      for i in range(device_count)
  ]).reshape(shape)


def create_device_list(shape, device_type):
  # [FUNCTION DOC]
  # HELPER: flat version of the device-spec grid builder.
  # Inputs: shape (mesh dimensions) and device_type (string). Returns a flat list of device
  # specifications in row-major order, as produced by the device array builder.
  devices = create_device_array(shape, device_type)
  return np.ravel(devices).tolist()


def reset_context():
  # [FUNCTION DOC]
  # HELPER: wrapper that resets the global eager runtime context (via a private API).
  # Takes no inputs and returns nothing. Side effect: discards the current eager context so a
  # fresh one is created on next use, which allows device configuration to be changed again.
  context._reset_context()  # pylint: disable=protected-access


def reset_logical_devices(device_type, count):
  # [FUNCTION DOC]
  # Re-partitions the physical devices of one type into the requested number of logical devices.
  # Inputs: device_type (string, CPU or GPU, case-insensitive) and count (integer, desired
  # number of logical devices). First resets the eager context. Raises a value error for any
  # device type other than CPU or GPU (note: the check happens after the reset). If count is
  # smaller than the number of physical devices, only the first count physical devices are made
  # visible. Then the logical devices are spread as evenly as possible across the physical
  # devices; GPU logical devices get a fixed memory limit (1 GB) and a device ordinal, CPU ones
  # use default settings. Returns nothing; mutates global device configuration.
  """Resets logical devices for CPU/GPU.

  Logical devices can only be instantiated once on a particular context. For
  now, context re-use is triggering some function duplication errors, so we
  reset the context on each call.

  Args:
    device_type: The device_type to reset.
    count: numbers of virtual device to reset to.
  """
  reset_context()
  devices = tf_config.list_physical_devices(device_type)
  if device_type.upper() not in ('CPU', 'GPU'):
    raise ValueError(
        'resetting logical device for non-supported device type : %s'
        % device_type
    )

  if count < len(devices):
    devices = devices[:count]
    tf_config.set_visible_devices(devices, device_type=device_type.upper())

  for i, device in enumerate(devices):
    n = (i + 1) * count // len(devices) - i * count // len(devices)
    assert n > 0  # guaranteed if count >= len(devices)
    configs = []
    for ordinal in range(n):
      if device_type.upper() == 'GPU':
        dev_config = context.LogicalDeviceConfiguration(
            memory_limit=_DEFAULT_GPU_MEMORY_LIMIT,
            experimental_device_ordinal=ordinal,
        )
      else:
        dev_config = context.LogicalDeviceConfiguration()
      configs.append(dev_config)

    tf_config.set_logical_device_configuration(device, configs)


def list_local_logical_devices(device_type):
  # [FUNCTION DOC]
  # Lists the logical devices of a given type that belong to this client. Input: device_type,
  # a string such as CPU, GPU or TPU. Returns a list of logical device descriptions, filtered
  # to those local to the current process (see the nested filter). Reads global device state.
  """Returns a list of local logial devices."""

  # When coordinator service is enabled, list_logical_devices returns
  # a global list.
  devices = tf_config.list_logical_devices(device_type)

  def is_local(device):
    # [FUNCTION DOC]
    # HELPER: filter for the local-device listing. Input: a logical device description.
    # Returns true if its job is unset or the local host job, or if its job and task match this
    # DTensor client's job name and client id; otherwise returns false.
    spec = tf_device.DeviceSpec.from_string(device.name)
    if spec.job is None or spec.job == 'localhost':
      return True
    elif spec.job == config.job_name() and spec.task == config.client_id():
      return True
    return False

  return [d for d in devices if is_local(d)]


def is_tfrt_enabled():
  # [FUNCTION DOC]
  # HELPER: thin wrapper over the eager context query. Takes no inputs; returns true if the
  # TFRT runtime is enabled in the current context, otherwise false.
  return context.is_tfrt_enabled()


FLAGS = flags.FLAGS


class DTensorBaseTest(tf_test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: common base class for DTensor unit tests.
  # Combines the TensorFlow test case with parameterized-test support. Holds a backend
  # configurator created per test in set-up and torn down in tear-down. Offers mesh
  # configuration, device-type and runtime skip helpers, and an assertion comparing a DTensor
  # against an expected local value and layout.
  """Provides comparison helper for dtensor vs local results."""

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level set-up. Only delegates to the parent implementation; adds no behavior.
    super(DTensorBaseTest, cls).setUpClass()

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test set-up. Calls the parent set-up and then creates the backend configurator bound to
    # this test, stored as an instance attribute. Returns nothing.
    super().setUp()
    self._backend_configurator = DTensorTestBackendConfigurator(self)

  def tearDown(self):
    # [FUNCTION DOC]
    # Per-test tear-down. Waits for all pending asynchronous eager operations to finish. In all
    # cases (even if the wait raises) it then resets the DTensor singleton device, tears down the
    # backend configurator (shutting down the accelerator system if initialized) and calls the
    # parent tear-down. Returns nothing.
    # Make sure all async ops finish.
    try:
      context.async_wait()
    finally:
      # TODO(hthu): Remove the reset once we fixed the CopyToMesh with
      # DefaultMesh placement issue.
      reset_dtensor()

      self._backend_configurator.tearDown()
      super().tearDown()

  @staticmethod
  def configTestMesh(  # pylint: disable=invalid-name
      device_type_mesh_map: typing.Dict[typing.Text, layout_lib.Mesh],
  ) -> layout_lib.Mesh:
    # [FUNCTION DOC]
    # Static method that picks and prepares the mesh matching the hardware available.
    # Input: device_type_mesh_map, a mapping from device type names (CPU, GPU, TPU) to meshes.
    # Resets the eager context, then: if a TPU is present uses the TPU mesh and initializes the
    # TPU accelerator system; else if physical GPUs exist uses the GPU mesh, splits the GPUs into
    # as many logical devices as the mesh size and initializes the GPU system; otherwise uses
    # the CPU mesh, creates that many logical CPUs and initializes the CPU system. Returns the
    # chosen mesh. Raises a value error if the mapping lacks the mesh for the selected type.
    # Side effects: resets global context and initializes accelerator system.
    """Configs corresponding mesh given test context.

    If runs on a CPU mesh, set virtual device on CPU.
    If runs on a GPU mesh, sets virtual device on GPU with proper memory limits.
    if runs on a TPU mesh, initializes TPU system.

    Args:
      device_type_mesh_map: A dictionary containing device_type -> mesh mapping.

    Returns:
      A properly configured mesh for use in test.
    """
    reset_context()

    def get_mesh(device_type):
      # [FUNCTION DOC]
      # HELPER: looks up the mesh for a device type in the enclosing mapping. Returns it, or
      # raises a value error stating that a mesh of that type is required if it is missing.
      mesh = device_type_mesh_map.get(device_type, None)
      if mesh is None:
        raise ValueError(
            'Requires a %s mesh to run test on %s.' % (device_type, device_type)
        )
      return mesh

    mesh = None
    if is_tpu_present():
      mesh = get_mesh('TPU')
      reset_context()
      accelerator_util.initialize_accelerator_system('TPU')
    elif tf_config.list_physical_devices('GPU'):
      mesh = get_mesh('GPU')
      reset_logical_devices('GPU', np.prod(mesh.shape()))
      accelerator_util.initialize_accelerator_system('GPU')
    else:
      mesh = get_mesh('CPU')
      reset_logical_devices('CPU', np.prod(mesh.shape()))
      accelerator_util.initialize_accelerator_system('CPU')

    return mesh

  def skipForDeviceType(  # pylint: disable=invalid-name
      self,
      device_type: typing.List[str],
      reason: str,
      unless_device_count_equals_to=None,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for GPU the test is skipped whenever a GPU is present, and the optional device
    # count is only honored for TPU; for CPU it skips only when the sole physical device type is
    # CPU. The docstring wording about the count matches this, but the name suggests an
    # unconditional per-type skip.
    # Skips the current test depending on which hardware is present. Inputs: device_type (list of
    # strings among CPU, GPU, TPU), reason (message for the skip) and the optional
    # unless_device_count_equals_to (integer, TPU only; skip unless that many local TPU logical
    # devices exist). Raises the skip signal via the test skip method (which also tears down
    # the backend). Returns nothing if no skip is triggered.
    """Skip the test for the specific device_type.

    Args:
      device_type: list of device types, one of "CPU", "GPU", or "TPU".
      reason: string that describe the reason for skipping the test.
      unless_device_count_equals_to: Optional int. This parameter only works if
        device_type is "TPU". If set, the test will be skipped unless the number
        of TPUs equals to the specified count.
    """
    physical_device_types = set(
        [d.device_type for d in tf_config.list_physical_devices()]
    )
    for device in device_type:
      if device == 'TPU' and is_tpu_present():
        if unless_device_count_equals_to is None:
          self.skipTest(reason)
        elif (
            len(list_local_logical_devices(device))
            != unless_device_count_equals_to
        ):
          self.skipTest(reason)
      if (
          device == 'CPU'
          and len(physical_device_types) == 1
          and 'CPU' in physical_device_types
      ):
        # Make sure we skip when only `CPU` is present.
        self.skipTest(reason)
      if device == 'GPU' and 'GPU' in physical_device_types:
        self.skipTest(reason)

  def skipForTfrt(self, reason: str):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Skips the current test with the given reason string if the TFRT runtime is enabled;
    # otherwise does nothing. Returns nothing.
    if is_tfrt_enabled():
      self.skipTest(reason)

  def skipTest(self, reason):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Override of the standard skip. Input: reason string. If the backend configurator already
    # exists (the skip may happen inside the parent set-up before it is created), tears the
    # backend down first, then delegates to the parent skip which raises the skip signal.
    # skipTest() may be called in super().setUp()
    if hasattr(self, '_backend_configurator'):
      self._backend_configurator.tearDown()
    super().skipTest(reason)

  def assertDTensorEqual(
      self,  # pylint: disable=invalid-name
      expected_result,
      expected_layout,
      result_dtensor,
      tol=DEFAULT_TOL,
  ):
    # [FUNCTION DOC]
    # Asserts that a DTensor has the expected value and, optionally, layout.
    # Inputs: expected_result (local numpy-like array), expected_layout (layout or nothing to
    # skip the layout check), result_dtensor (the DTensor or resource variable under test) and
    # tol (absolute and relative tolerance for the value comparison, default 1e-5).
    # Steps: a resource variable is replaced by its value; if a layout is given, the fetched
    # layout must equal it (the failure message prints both layouts, trimming an identical mesh
    # part); the global shape must match; the DTensor is converted to a numpy array whose dtype
    # must equal the expected dtype; the values must be close within tol; finally every device
    # component along replicated mesh dimensions must match the component at index zero of that
    # dimension (compared with default tolerance, ignoring tol).
    # Fails the test via assertions; returns nothing.
    """Asserts DTensor is of the particular value."""
    if issubclass(
        type(result_dtensor), resource_variable_ops.BaseResourceVariable
    ):
      result_dtensor = result_dtensor.value()
    if expected_layout is not None:
      # This, the assertEqual, is a pure proto raw bytes comparison. To make it
      # human-readable, use the `to_string` api for Layout for the dedicated msg
      # field.
      #
      # Futhurmore, as the mesh part is very long and usually identical. Try to
      # cut them as well, to make it easier to read.
      expected_str = expected_layout.to_string()
      got_str = api.fetch_layout(result_dtensor).to_string()
      index_for_mesh = expected_str.find('mesh:')
      if (
          index_for_mesh != -1
          and got_str.find(expected_str[index_for_mesh:]) != -1
      ):
        # the mesh part is same. cut them so it is more readable.
        expected_str = expected_str[:index_for_mesh]
        got_str = got_str[: got_str.find('mesh:')]

      self.assertEqual(
          api.fetch_layout(result_dtensor),
          expected_layout,
          msg=(
              '=======\nexpected layout is\n  {}\n\nwhile got layout is\n  {}\n'
              .format(expected_str, got_str)
          ),
      )

    layout = api.fetch_layout(result_dtensor)
    unpacked = [t.numpy() for t in api.unpack(result_dtensor)]

    # Check global shape.
    self.assertAllEqual(expected_result.shape, result_dtensor.shape)

    result_dtensor = numpy_util.to_numpy(result_dtensor)

    # Check dtype.
    # Note: This check needs be after result_dtensor is converted
    # into numpy, due to failure with Numpy version 1.18.5.
    self.assertEqual(
        expected_result.dtype, result_dtensor.dtype, result_dtensor
    )

    # Check value on concatenated result DTensor.
    self.assertAllClose(expected_result, result_dtensor, atol=tol, rtol=tol)

    # In addition to check the 'concatenated' DTensor, we also check all
    # "replicated" parts are same.
    #
    # The algorithm is simple:
    # 1. For a mesh with topology (x,y,z,p), and a DTensor with layout ('',z,x).
    # 2. Create some data structures:
    #   - create a mapping from device id (called offset below) to mesh
    #     location. For the mesh above, loc {x:1,y:2,z:2,p:0} means the device
    #     is located at that coordinates in the 4-D mesh.
    #   - create a mapping from mesh location to device id.
    # 3. Find all replicated mesh dimension names, i.e., 'y' and `p` in the
    #     example above.
    # 4. Iterate over all unpacked components, translate the offset (device id)
    #    to mesh location, called (x',y',z',p').
    #   - For `y`, which is replicated dim in the mesh, check all unpacked
    #     components at (x',*,z',p') are same as the component at (x',0,z',p').
    #   - For `p`, which is also replicated dim in the mesh, check all unpacked
    #     components at (x',y',z',*) are same as the component at (x',y',z',0).

    def hash_key(loc):
      # [FUNCTION DOC]
      # HELPER: turns a mesh location (mapping from dimension names to integer coordinates) into
      # a string usable as a lookup key: the entries are sorted by name and serialized as text.
      """Hash key for Python dict."""
      # Python dict is unhashable. Creates a sorted dict and dumps as json str.
      d = collections.OrderedDict(sorted(loc.items(), key=lambda x: x[0]))
      return json.dumps(d)

    offset_to_mesh_loc_dict = layout.mesh.unravel_index()
    mesh_loc_to_offset_dict = {}
    for offset, loc in offset_to_mesh_loc_dict.items():
      mesh_loc_to_offset_dict[hash_key(loc)] = offset

    # pylint: disable=protected-access
    replicated_dims = [
        x for x in layout.mesh.dim_names if x not in layout.sharding_specs
    ]
    # pylint: enable=protected-access

    for offset, tensor in enumerate(unpacked):
      mesh_loc = offset_to_mesh_loc_dict[offset]
      for dim_sharding in replicated_dims:
        if mesh_loc[dim_sharding] != 0:
          mesh_loc = copy.deepcopy(mesh_loc)  # deepcopy as we will mutate
          mesh_loc[dim_sharding] = 0
          offset = mesh_loc_to_offset_dict[hash_key(mesh_loc)]
          # tol is be as low as possible as they should match "exactly". so, we
          # ignore the `tol` passed by caller and choose the default one.
          self.assertAllClose(tensor, unpacked[offset])


def product(*lists):
  # [FUNCTION DOC]
  # HELPER: builds named parameter combinations for parameterized tests.
  # Input: any number of lists, each containing tuples whose first item is a name fragment and
  # remaining items are parameter values. Returns a list with one tuple per element of the
  # Cartesian product: the concatenated name fragments followed by all parameter values in order.
  """Makes a product of names parameters list."""
  # Each element lists should be a tuple of tuples of the form
  # (("test1", ...), ("test2", ...), ...).
  # Function returns the product of the lists with the labels concatenated.
  return [  # pylint: disable=g-complex-comprehension
      (''.join(p[0] for p in elt), *sum((p[1:] for p in elt), ()))
      for elt in itertools.product(*lists)
  ]


def reset_dtensor():
  # [FUNCTION DOC]
  # HELPER: test-only wrapper that resets the singleton DTensor device through a private API.
  # Takes no inputs, returns nothing; global DTensor device state is cleared.
  """Resets the singleton DTensor Device.

  This behavior is not generally exposed and only meant to be used in tests.
  """
  api._reset()  # pylint: disable=protected-access


__all__ = [
    'DEFAULT_TOL',
    'DTensorTestUtilBackend',
    'DTENSOR_TEST_UTIL_BACKEND',
    'create_device_ids_array',
    'create_device_array',
    'create_device_list',
    'reset_context',
    'reset_logical_devices',
    'list_local_logical_devices',
    'is_tfrt_enabled',
    'FLAGS',
    'DTensorBaseTest',
    'product',
    'reset_dtensor',
    'is_tpu_present',
    'is_gpu_present',
    'use_multi_device_mode',
]
