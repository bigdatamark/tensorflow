# [MODULE DOC]
# Tests for the packed distributed variable module: packing variables across two virtual CPUs,
# per-device reads and updates in eager and function mode, save and restore, export of a
# trackable holding a packed variable, the device-pinned wrapper, and absence of reference-cycle
# garbage. Runs with eager execution enabled when executed as a script.
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
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import packed_distributed_variable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable


class TestExportArchive(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # HELPER: fixture for the export test.
  # Trackable object holding a packed variable built from two variables on virtual CPUs 0 and 1,
  # a function wrapping its update method, and a method to save itself as a saved model.

  def __init__(self):
    # [FUNCTION DOC]
    # Creates one variable on each of two virtual CPUs (initial values 1.0 and 2.0), packs them,
    # and wraps the update method as a compiled function. Returns nothing.
    with ops.device('/cpu:0'):
      v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
    with ops.device('/cpu:1'):
      v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')

    self._packed_var = packed_distributed_variable.PackedDistributedVariable(
        [v0, v1]
    )
    self._fn = def_function.function(self.update_var)

  @def_function.function
  def update_var(self):
    # [FUNCTION DOC]
    # Compiled function: adds 3.0 to the packed variable then subtracts 1.0 from the result.
    # Returns nothing.
    self._packed_var.assign_add(3.0).assign_sub(1.0)

  def save_function(self, directory):
    # [FUNCTION DOC]
    # Saves this object as a saved model into the given directory. Writes files; returns nothing.
    save.save(self, directory)


class PackedDistributedVariableTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the packed distributed variable and the device-pinned wrapper. The
  # shared fixture splits the first physical CPU into two logical CPUs.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the base setup, then configures the first physical CPU as two logical devices.
    super(PackedDistributedVariableTest, self).setUp()
    cpus = config.list_physical_devices('CPU')
    # Set 2 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
    ])

  def testPackedVariable(self):
    # [FUNCTION DOC]
    # Packs variables on two CPUs (values 1.0 and 2.0). Eagerly, the packed handle is not packed
    # and the initialized check is truthy. On each device, the current-device variable matches
    # the original, and assign 2.0 then add 1.0 yields 3.0. Inside a function, the handle is
    # packed; the updates on the two devices read back 5.0 and -3.0 respectively.
    with ops.device('/cpu:0'):
      v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
    with ops.device('/cpu:1'):
      v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
    self.assertFalse(packed_var.handle.is_packed)
    self.assertTrue(packed_var.is_initialized)

    with ops.device('/cpu:0'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v0)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    with ops.device('/cpu:1'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v1)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    @def_function.function
    def update_var():
      # [FUNCTION DOC]
      # Nested function: checks the handle is packed, then on CPU 0 adds 3 and subtracts 1 and
      # reads; on CPU 1 subtracts 4 and 2 and reads. Returns both read values.
      self.assertTrue(packed_var.handle.is_packed)
      with ops.device('/cpu:0'):
        packed_var.assign_add(3.0).assign_sub(1.0)
        read0 = packed_var.value()
      with ops.device('/cpu:1'):
        packed_var.assign_sub(4.0).assign_sub(2.0)
        read1 = packed_var.value()

      return read0, read1

    self.assertAllEqual(update_var(), (5.0, -3.0))

  def testPackedVariableSaveAndRestore(self):
    # [FUNCTION DOC]
    # Packs two variables, updates each device's copy to 3.0, saves the packed variable to a
    # temporary directory, loads it back, and asserts the restored value equals the original.
    with ops.device('/cpu:0'):
      v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
    with ops.device('/cpu:1'):
      v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])

    with ops.device('/cpu:0'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v0)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    with ops.device('/cpu:1'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v1)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    export_dir = self.get_temp_dir()
    save.save(packed_var, export_dir)

    restored_packed_var = load.load(export_dir)
    self.assertAllEqual(restored_packed_var, packed_var)

  def testPackedVariableExport(self):
    # [FUNCTION DOC]
    # Builds the export fixture and saves it to a temporary directory. Only asserts that saving
    # does not raise; the restore step is commented out.
    export_dir = self.get_temp_dir()
    export_archive = TestExportArchive()
    export_archive.save_function(export_dir)

    # restored_packed_var = load.load(export_dir)
    # logging.info(f'Restored var {restored_packed_var} ')

  def testPackedVarAndDevice(self):
    # [FUNCTION DOC]
    # Packs variables (1.0 on CPU 0, 2.0 on CPU 1). Wrapper for device 0 has an unpacked handle
    # and multiplying it by 2 gives 2.0. Wrapper for device 1 assigns 3.0 and returns 3.0. In a
    # function (handle packed), adding 3 on device 0 and reading gives 4.0, and adding 2 to the
    # device 1 wrapper gives 5.0.
    device0 = device_util.canonicalize('/cpu:0')
    device1 = device_util.canonicalize('/cpu:1')

    with ops.device(device0):
      v0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device(device1):
      v1 = resource_variable_ops.ResourceVariable(2.0)

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])

    packed_var0 = packed_distributed_variable.PackedVarAndDevice(
        packed_var, device0)
    self.assertFalse(packed_var0.handle.is_packed)
    self.assertAllEqual(math_ops.mul(packed_var0, 2.0), 2.0)

    packed_var1 = packed_distributed_variable.PackedVarAndDevice(
        packed_var, device1)
    self.assertAllEqual(packed_var1.assign(3.0), 3.0)

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # Nested function: asserts the handle is packed, adds 3 through the device 0 wrapper, and
      # returns that wrapper's value with the device 1 wrapper plus 2.
      self.assertTrue(packed_var.handle.is_packed)
      var0 = packed_distributed_variable.PackedVarAndDevice(packed_var, device0)
      var0.assign_add(3.0)
      var1 = packed_distributed_variable.PackedVarAndDevice(packed_var, device1)
      return var0.value(), math_ops.add(var1, 2.0)

    self.assertAllEqual(func(), (4.0, 5.0))

  @test_util.assert_no_garbage_created
  def testNoGarbage(self):
    # [FUNCTION DOC]
    # Packs two variables, then looks up a nonexistent attribute on a device wrapper; the
    # decorator asserts no uncollectable garbage (reference cycles) is created.
    device0 = device_util.canonicalize('/cpu:0')
    device1 = device_util.canonicalize('/cpu:1')

    with ops.device(device0):
      v0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device(device1):
      v1 = resource_variable_ops.ResourceVariable(2.0)

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
    # This needs a workaround to avoid creating reference cycles if the
    # attribute doesn't exist.
    hasattr(packed_var.on_device('/cpu:0'), 'nonexist')


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
