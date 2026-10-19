# [MODULE DOC]
# Unit tests for the compat module: forward_compatible, the horizon context manager (including
# restoration after errors) and the environment variable that shifts the compatibility date.
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
"""Tests for forward and backwards compatibility utilities."""

import datetime
import os
from tensorflow.python.compat import compat
from tensorflow.python.platform import test


class CompatTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the forward compatibility utilities, with helpers that derive dates relative
  # to the module's horizon date.

  def _compatibility_date(self):
    # [FUNCTION DOC]
    # HELPER: returns the module's horizon date as a (year, month, day) triple.
    date = compat._FORWARD_COMPATIBILITY_HORIZON  # pylint: disable=protected-access
    return (date.year, date.month, date.day)

  def _n_days_after(self, n):
    # [FUNCTION DOC]
    # HELPER: returns the (year, month, day) triple of the horizon date shifted by n days
    # (integer, may be negative).
    date = compat._FORWARD_COMPATIBILITY_HORIZON + datetime.timedelta(days=n)  # pylint: disable=protected-access
    return (date.year, date.month, date.day)

  def test_basic(self):
    # [FUNCTION DOC]
    # Asserts the day before the horizon is already compatible and the horizon date itself is not.
    compatibility_date = self._compatibility_date()
    one_day_before = self._n_days_after(-1)
    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertFalse(compat.forward_compatible(*compatibility_date))

  def test_past(self):
    # [FUNCTION DOC]
    # Sets the horizon to a past date (ignored with a warning) and asserts a future-ish date
    # (2020) is still reported compatible.
    with compat.forward_compatibility_horizon(2018, 9, 18):
      self.assertTrue(compat.forward_compatible(2020, 4, 4))

  def test_decorator(self):
    # [FUNCTION DOC]
    # Inside the horizon context set to one day after, asserts the horizon date is compatible but
    # the new date is not; after leaving the context asserts the horizon date is not compatible
    # again (value restored).
    compatibility_date = self._compatibility_date()
    one_day_after = self._n_days_after(1)
    with compat.forward_compatibility_horizon(*one_day_after):
      self.assertTrue(compat.forward_compatible(*compatibility_date))
      self.assertFalse(compat.forward_compatible(*one_day_after))

    # After exiting context manager, value should be reset.
    self.assertFalse(compat.forward_compatible(*compatibility_date))

  def test_decorator_with_failure(self):
    # [FUNCTION DOC]
    # Raises a custom error inside the horizon context, swallows it, then asserts the date was
    # restored to the default.
    compatibility_date = self._compatibility_date()
    one_day_after = self._n_days_after(1)

    class DummyError(Exception):
      # [CLASS DOC]
      # HELPER: throwaway exception type raised to test that the context manager restores state.
      pass

    try:
      with compat.forward_compatibility_horizon(*one_day_after):
        raise DummyError()
    except DummyError:
      pass  # silence DummyError

    # After exiting context manager, value should be reset.
    self.assertFalse(compat.forward_compatible(*compatibility_date))

  def test_environment_override(self):
    # [FUNCTION DOC]
    # Checks the days-delta environment variable: with it unset, with it set to ten followed by a
    # recomputation, after removing it, and combined with the horizon context manager; asserts
    # compatibility results at various offsets each time. Cleanup removes the variable.
    var_name = 'TF_FORWARD_COMPATIBILITY_DELTA_DAYS'

    def remove_os_environment_var():
      # [FUNCTION DOC]
      # HELPER: cleanup callback that deletes the environment variable, ignoring it if missing.
      try:
        del os.environ[var_name]
      except KeyError:
        pass

    self.addCleanup(remove_os_environment_var)

    compatibility_date = self._compatibility_date()
    one_day_before = self._n_days_after(-1)
    one_day_after = self._n_days_after(1)
    ten_days_after = self._n_days_after(10)
    nine_days_after = self._n_days_after(9)

    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertFalse(compat.forward_compatible(*compatibility_date))
    self.assertFalse(compat.forward_compatible(*one_day_after))
    self.assertFalse(compat.forward_compatible(*nine_days_after))
    self.assertFalse(compat.forward_compatible(*ten_days_after))

    os.environ[var_name] = '10'
    compat._update_forward_compatibility_date_number()
    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertTrue(compat.forward_compatible(*compatibility_date))
    self.assertTrue(compat.forward_compatible(*one_day_after))
    self.assertTrue(compat.forward_compatible(*nine_days_after))
    self.assertFalse(compat.forward_compatible(*ten_days_after))

    del os.environ[var_name]
    compat._update_forward_compatibility_date_number()
    self.assertTrue(compat.forward_compatible(*one_day_before))
    self.assertFalse(compat.forward_compatible(*compatibility_date))
    self.assertFalse(compat.forward_compatible(*one_day_after))
    self.assertFalse(compat.forward_compatible(*nine_days_after))
    self.assertFalse(compat.forward_compatible(*ten_days_after))

    # Now test interaction between environment variable and context func.
    os.environ[var_name] = '10'
    compat._update_forward_compatibility_date_number()
    self.assertTrue(compat.forward_compatible(*one_day_after))
    with compat.forward_compatibility_horizon(*one_day_after):
      self.assertTrue(compat.forward_compatible(*one_day_before))
      self.assertTrue(compat.forward_compatible(*compatibility_date))
      self.assertFalse(compat.forward_compatible(*one_day_after))
      self.assertFalse(compat.forward_compatible(*nine_days_after))
      self.assertFalse(compat.forward_compatible(*ten_days_after))
    self.assertTrue(compat.forward_compatible(*one_day_after))


if __name__ == '__main__':
  test.main()
