# [MODULE DOC]
# Tests for MemoryChecker: verifies that snapshot-based leak assertions pass for stable or
# near-stable allocation patterns, raise for steady leaks, and that the new-Python-object assertion
# reports growth. Runs with eager execution enabled when executed as a script.
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
# =============================================================================

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.memory_checker import MemoryChecker
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class MemoryCheckerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for MemoryChecker, run with eager-op-as-function behaviour via a decorator. Each test
  # opens a checker context, records snapshots around controlled allocations of constant tensors,
  # and checks the leak or new-object assertions. No shared fixtures.

  def testNoLeakEmpty(self):
    # [FUNCTION DOC]
    # Records four snapshots with no allocations in between, generates the report, and asserts the
    # leak check passes (raises nothing).
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testNoLeak1(self):
    # [FUNCTION DOC]
    # Records a snapshot, allocates one constant tensor kept alive, then records three more
    # snapshots. A single allocation is the permitted one-snapshot exception, so after generating
    # the report the leak assertion must pass.
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      x = constant_op.constant(1)  # pylint: disable=unused-variable
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testNoLeak3(self):
    # [FUNCTION DOC]
    # Over ten iterations appends a constant tensor to a retained list except at iterations five
    # and seven, recording a snapshot each iteration. The allocation is missing at two snapshots,
    # so it is not a consistent leak; after the report the leak assertion must pass.
    with MemoryChecker() as memory_checker:
      tensors = []
      for i in range(10):
        if i not in (5, 7):
          tensors.append(constant_op.constant(1))
        memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testLeak1(self):
    # [FUNCTION DOC]
    # Records snapshots with a tensor allocated (and kept) between the first and second snapshot and
    # another between the second and third, so allocations occur at two snapshots. After the report,
    # asserts the leak check raises an assertion error.
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      x = constant_op.constant(1)  # pylint: disable=unused-variable
      memory_checker.record_snapshot()
      y = constant_op.constant(1)  # pylint: disable=unused-variable
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.report()
    with self.assertRaises(AssertionError):
      memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testLeak3(self):
    # [FUNCTION DOC]
    # Over ten iterations appends a constant tensor to a retained list and records a snapshot each
    # time, a consistent leak. After the report, asserts the leak check raises an assertion error.
    with MemoryChecker() as memory_checker:
      tensors = []
      for _ in range(10):
        tensors.append(constant_op.constant(1))
        memory_checker.record_snapshot()

    memory_checker.report()
    with self.assertRaises(AssertionError):
      memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testNoNewPythonObjectsEmpty(self):
    # [FUNCTION DOC]
    # Records two snapshots with nothing allocated between them and asserts the new-Python-objects
    # check passes without raising.
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.assert_no_new_python_objects()

  def testNewPythonObjects(self):
    # [FUNCTION DOC]
    # Records a snapshot, creates a constant tensor kept alive, records a second snapshot, and
    # asserts the new-Python-objects check raises an assertion error whose message mentions new
    # Python objects. Finally asserts the tensor is not null, only to keep it referenced so it is
    # not optimized away.
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      x = constant_op.constant(1)
      memory_checker.record_snapshot()

    with self.assertRaisesRegex(AssertionError, 'New Python objects'):
      memory_checker.assert_no_new_python_objects()

    # use x to avoid any potential for optimizing it away.
    self.assertIsNot(x, None)

  def testNewPythonObjectBelowThreshold(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests a threshold is exercised, but no threshold is passed; the test
    # actually checks that a short-lived object deleted before the second snapshot is not reported.
    # Defines a local empty class, records a snapshot, creates and immediately deletes an instance,
    # records another snapshot, and asserts the new-Python-objects check passes without raising.

    class Foo(object):
      # [CLASS DOC]
      # Empty local class whose instance is created and deleted to exercise the object counting.
      pass

    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      foo = Foo()
      del foo
      memory_checker.record_snapshot()

    memory_checker.assert_no_new_python_objects()


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
