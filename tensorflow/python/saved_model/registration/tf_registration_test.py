# [MODULE DOC]
# Test module guarding the allowlists of TensorFlow's saved-model registrations. It compares the
# names registered at import time (serializable classes and checkpoint savers) with two text files
# shipped as data (tf_serializable_allowlist.txt and tf_checkpoint_saver_allowlist.txt), so any
# added or removed registration must be approved by the Saved Model team.
# Import-time side effects: touches the TPU embedding class and the save function so their modules
# load and register, registers a dummy serializable class and two dummy checkpoint savers that are
# deliberately absent from the allowlists, and defines the allowlist path constants.
# Run as a test binary via the platform test main entry point.
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
"""Tests TensorFlow registered objects.

New or deleted registrations must be approved by the Saved Model team.
"""
import os

import tensorflow as tf

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model.registration import test_util

SERIALIZABLE_ALLOWLIST = os.path.join(resource_loader.get_data_files_path(),
                                      "tf_serializable_allowlist.txt")
CHECKPOINT_SAVER_ALLOWLIST = os.path.join(resource_loader.get_data_files_path(),
                                          "tf_checkpoint_saver_allowlist.txt")

# call TPUEmbedding to load the file and run its registration
_ = tf.tpu.experimental.embedding.TPUEmbedding
# call save to load the file and run its registration
_ = tf.saved_model.save


@registration.register_tf_serializable()
class NotInAllowlistExample(tf.Variable):
  # [CLASS DOC]
  # HELPER: sentinel fixture for AllowlistTest.
  # Empty variable subclass registered as a serializable class under the tf namespace. It is
  # intentionally missing from the allowlist so the test can verify that unlisted registrations
  # are detected. It has no state or behavior of its own.
  pass


registration.register_tf_checkpoint_saver(
    name="TestDummySaver",
    predicate=lambda: False,
    save_fn=lambda: None,
    restore_fn=lambda: None)

registration.register_tf_checkpoint_saver(
    name="NotInAllowlistExample",
    predicate=lambda: False,
    save_fn=lambda: None,
    restore_fn=lambda: None)


class AllowlistTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that the set of registered checkpoint savers and registered serializable
  # classes exactly matches the corresponding allowlist data files. Holds no fixtures; each test
  # delegates to the shared allowlist comparison helper.

  def _test_allowlist(self, allowlist_file, registrations):
    # [FUNCTION DOC]
    # HELPER: shared comparison logic for the allowlist tests in AllowlistTest.
    # Reads the allowlist file (path given by allowlist_file), ignoring blank lines and lines
    # starting with a hash, and compares it to registrations (an iterable of registered names).
    # First it asserts that the sentinel name for the unlisted example class is reported as
    # missing from the allowlist, then removes it from consideration. It then builds a message
    # listing names registered but not allowlisted, and names allowlisted but no longer
    # registered. Returns nothing. Raises an assertion error with instructions to update the
    # allowlist and obtain owner approval if either difference is non-empty; otherwise passes.
    # The sentinel assertion itself fails if the sentinel is not reported missing.
    allowlist = set([
        s.strip()
        for s in file_io.read_file_to_string(allowlist_file).splitlines()
        if s.strip() and not s.startswith("#")
    ])
    registered_names = set(registrations)

    missing_from_allowlist = registered_names - allowlist
    self.assertIn("tf.NotInAllowlistExample", missing_from_allowlist)
    missing_from_allowlist.remove("tf.NotInAllowlistExample")

    if missing_from_allowlist:
      msg = ("[NEEDS ATTENTION] Registered names found that were not added to "
             "the allowlist. Add the following names to the list:\n\t" +
             "\n\t".join(missing_from_allowlist))
    else:
      msg = "[OK] All registered names have been added to the allowlist.  ✓"

    msg += "\n\n"

    missing_registered_names = allowlist - registered_names
    if missing_registered_names:
      msg += ("[NEEDS ATTENTION] Some names were found in the allowlist that "
              "are not registered in TensorFlow. This could mean that a "
              "registration was removed from the codebase. If this was "
              "intended, please remove the following from the allowlist:\n\t" +
              "\n\t".join(missing_registered_names))
    else:
      msg += ("[OK] All allowlisted names are registered in the Tensorflow "
              "library. ✓")

    if missing_from_allowlist or missing_registered_names:
      raise AssertionError(
          "Error found in the registration allowlist.\nPlease update the "
          "allowlist at .../tensorflow/python/saved_model/registration/"
          f"{os.path.basename(allowlist_file)}.\n\n" + msg +
          "\n\nAfter making changes, request approval from "
          " tf-saved-model-owners@.")

  def test_checkpoint_savers(self):
    # [FUNCTION DOC]
    # Test: compares all registered checkpoint savers, obtained from the registration test
    # utilities, against the checkpoint saver allowlist file. Expects no differences apart from
    # the deliberately unlisted sentinel saver.
    self._test_allowlist(CHECKPOINT_SAVER_ALLOWLIST,
                         test_util.get_all_registered_checkpoint_savers())

  def test_serializables(self):
    # [FUNCTION DOC]
    # Test: compares all registered serializable classes, obtained from the registration test
    # utilities, against the serializable allowlist file. Expects no differences apart from the
    # deliberately unlisted sentinel class.
    self._test_allowlist(SERIALIZABLE_ALLOWLIST,
                         test_util.get_all_registered_serializables())


if __name__ == "__main__":
  test.main()
