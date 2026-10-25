# [MODULE DOC]
# Tests for SavedModel fingerprinting: checks that saving writes a fingerprint file with the
# expected hashes, that the fingerprint reading API, the singleprint computation and conversion
# from and to protocol buffers behave correctly, including their error cases. Contains a single
# test class with small model-building helpers. Run through the standard TensorFlow test main.
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
"""Tests for SavedModel fingerprinting.

These tests verify that fingerprint is written correctly and that APIs for
reading it are correct.
"""
import os
import shutil

from tensorflow.core.config import flags
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.trackable import autotrackable


class FingerprintingTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the fingerprinting module and the fingerprint file written by save.
  # Helper methods create temporary saved models, trackable models with functions and read the
  # fingerprint proto from a file; setUp enables the saved model fingerprinting flag.

  def _create_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: creates an empty saved model for tests.
    # Saves an empty trackable object into a saved_model folder under the test temp directory,
    # registers cleanup to delete the folder after the test, and returns the directory path.
    root = autotrackable.AutoTrackable()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    self.addCleanup(shutil.rmtree, save_dir)
    return save_dir

  def _create_model_with_function(self):
    # [FUNCTION DOC]
    # HELPER: returns a trackable object with a function attribute f that doubles its input and
    # has no input signature. Nothing is saved.
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    return root

  def _create_model_with_input_signature(self):
    # [FUNCTION DOC]
    # HELPER: returns a trackable object whose function attribute f doubles its input and has a
    # fixed input signature of a float32 tensor of unknown shape. Nothing is saved.
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    return root

  def _create_model_with_data(self):
    # [FUNCTION DOC]
    # HELPER: returns a trackable object holding a float32 constant 1.0 as attribute x and a
    # function f with a float32 input signature that multiplies its input by x. Nothing is saved.
    root = autotrackable.AutoTrackable()
    root.x = constant_op.constant(1.0, dtype=dtypes.float32)
    root.f = def_function.function(
        lambda x: root.x * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    return root

  def _read_fingerprint(self, filename):
    # [FUNCTION DOC]
    # HELPER: loads a fingerprint file for assertions.
    # Reads the binary file at the given path and parses it into a fingerprint definition message
    # which is returned. Raises if the file cannot be read or parsed.
    fingerprint_def = fingerprint_pb2.FingerprintDef()
    with file_io.FileIO(filename, "rb") as f:
      fingerprint_def.ParseFromString(f.read())
    return fingerprint_def

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setup and resets the saved model fingerprinting configuration
    # flag to enabled (true) so that saving writes fingerprints. Side effect: global flag state.
    super().setUp()
    flags.config().saved_model_fingerprinting.reset(True)

  def test_basic_module(self):
    # [FUNCTION DOC]
    # Scenario: save an empty trackable object. Expected: the directory holds four files
    # including the fingerprint file; in the parsed fingerprint, the checksum and checkpoint hash
    # are positive (non-deterministic so exact values are not checked), the graph def program
    # hash and signature def hash equal fixed known values and the saved object graph hash is zero.
    save_dir = self._create_saved_model()
    files = file_io.list_directory_v2(save_dir)

    self.assertLen(files, 4)
    self.assertIn(constants.FINGERPRINT_FILENAME, files)

    fingerprint_def = self._read_fingerprint(
        file_io.join(save_dir, constants.FINGERPRINT_FILENAME))

    # We cannot check this value due to non-determinism in saving.
    self.assertGreater(fingerprint_def.saved_model_checksum, 0)
    self.assertEqual(fingerprint_def.graph_def_program_hash,
                     14830488309055091319)
    self.assertEqual(fingerprint_def.signature_def_hash, 12089566276354592893)
    self.assertEqual(fingerprint_def.saved_object_graph_hash, 0)
    # TODO(b/242348400): The checkpoint hash is non-deterministic, so we cannot
    # check its value here.
    self.assertGreater(fingerprint_def.checkpoint_hash, 0)

  def test_model_saved_with_different_signature_options(self):
    # [FUNCTION DOC]
    # Scenario: save one model three ways: with an explicit signature, without signatures, and a
    # model defined with an input signature. Expected: the signature-less save has a different
    # signature def hash than the explicit-signature save, while the explicit-signature save and
    # the input-signature save have equal graph def program hashes and signature def hashes.
    model = self._create_model_with_function()
    # Save the model with signatures specified in SaveOptions.
    sig_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(
        model,
        sig_dir,
        signatures=model.f.get_concrete_function(
            tensor_spec.TensorSpec(None, dtypes.float32)))
    # Save the model without signatures.
    no_sig_dir = os.path.join(self.get_temp_dir(), "saved_model2")
    save.save(model, no_sig_dir)
    # Save the model with an input signature specified.
    input_sig_dir = os.path.join(self.get_temp_dir(), "saved_model3")
    save.save(self._create_model_with_input_signature(), input_sig_dir)

    fingerprint_sig = self._read_fingerprint(
        file_io.join(sig_dir, constants.FINGERPRINT_FILENAME))
    fingerprint_no_sig = self._read_fingerprint(
        file_io.join(no_sig_dir, constants.FINGERPRINT_FILENAME))
    fingerprint_input_sig = self._read_fingerprint(
        file_io.join(input_sig_dir, constants.FINGERPRINT_FILENAME))

    # Check that the model saved with different options has different
    # SignatureDef hashes.
    self.assertNotEqual(fingerprint_sig.signature_def_hash,
                        fingerprint_no_sig.signature_def_hash)
    # Check that the model saved with the same concrete function has the same
    # regularized hashes.
    self.assertEqual(fingerprint_sig.graph_def_program_hash,
                     fingerprint_input_sig.graph_def_program_hash)
    self.assertEqual(fingerprint_sig.signature_def_hash,
                     fingerprint_input_sig.signature_def_hash)

  def test_read_fingerprint_api(self):
    # [FUNCTION DOC]
    # Scenario: save an empty model and read its fingerprint through the public reader and
    # directly from the proto file. Expected: the two compare equal.
    save_dir = self._create_saved_model()
    fingerprint = fingerprinting.read_fingerprint(save_dir)

    fingerprint_def = self._read_fingerprint(
        file_io.join(save_dir, constants.FINGERPRINT_FILENAME))

    self.assertEqual(fingerprint, fingerprint_def)

  def test_read_fingerprint_file_not_found(self):
    # [FUNCTION DOC]
    # Scenario: read the fingerprint of a nonexistent directory. Expected: a file-not-found error
    # whose message mentions a SavedModel Fingerprint Error.
    with self.assertRaisesRegex(FileNotFoundError,
                                "SavedModel Fingerprint Error"):
      fingerprinting.read_fingerprint("foo")

  def test_write_fingerprint(self):
    # [FUNCTION DOC]
    # Scenario: save a model with data via the save-and-return-nodes API while skipping the
    # checkpoint, then read the fingerprint. Expected: positive checksum, fixed known graph def
    # program hash and signature def hash, and a checkpoint hash of zero since no checkpoint
    # was written.
    save_dir = os.path.join(self.get_temp_dir(), "model_and_fingerprint")
    save.save_and_return_nodes(
        self._create_model_with_data(), save_dir,
        experimental_skip_checkpoint=True)  # checkpoint data won't be loaded

    fingerprint_def = fingerprinting.read_fingerprint(save_dir)

    # We cannot check this value due to non-determinism in serialization.
    self.assertGreater(fingerprint_def.saved_model_checksum, 0)
    self.assertEqual(fingerprint_def.graph_def_program_hash,
                     8947653168630125217)
    self.assertEqual(fingerprint_def.signature_def_hash, 15354238402988963670)
    self.assertEqual(fingerprint_def.checkpoint_hash, 0)

  def test_valid_singleprint(self):
    # [FUNCTION DOC]
    # Scenario: save a model with data, read its fingerprint and compute the singleprint.
    # Expected: the result matches a pattern made of the known graph def program hash, signature
    # def hash and saved object graph hash joined by slashes (the checkpoint hash is
    # non-deterministic and not matched).
    save_dir = os.path.join(self.get_temp_dir(), "singleprint_model")
    save.save(self._create_model_with_data(), save_dir)
    fingerprint = fingerprinting.read_fingerprint(save_dir)
    singleprint = fingerprint.singleprint()

    # checkpoint_hash is non-deterministic and not included
    self.assertRegex(singleprint,
                     "/".join(["8947653168630125217",  # graph_def_program_hash
                               "15354238402988963670",  # signature_def_hash
                               "1613952301283913051"  # saved_object_graph_hash
                               ]))

  def test_invalid_singleprint(self):
    # [FUNCTION DOC]
    # Scenario: build a Fingerprint with all default (null) values and ask for its singleprint.
    # Expected: a value error whose message mentions invalid fingerprint values.
    fingerprint = fingerprinting.Fingerprint()
    with self.assertRaisesRegex(ValueError,
                                "Encounted invalid fingerprint values"):
      fingerprint.singleprint()

  def test_valid_from_proto(self):
    # [FUNCTION DOC]
    # Scenario: save a model, read the raw fingerprint through the native layer, parse it into a
    # message and build a Fingerprint from it. Expected: the Fingerprint equals the message.
    save_dir = os.path.join(self.get_temp_dir(), "from_proto_model")
    save.save(self._create_model_with_data(), save_dir)
    fingerprint_def = fingerprint_pb2.FingerprintDef().FromString(
        fingerprinting_pywrap.ReadSavedModelFingerprint(save_dir))
    fingerprint = fingerprinting.Fingerprint.from_proto(fingerprint_def)
    self.assertEqual(fingerprint, fingerprint_def)

  def test_invalid_from_proto(self):
    # [FUNCTION DOC]
    # Scenario: save a model, then try building a Fingerprint from a saved model message (a wrong
    # message type). Expected: a value error saying the proto could not be deserialized.
    save_dir = os.path.join(self.get_temp_dir(), "from_proto_model")
    save.save(self._create_model_with_data(), save_dir)
    wrong_def = saved_model_pb2.SavedModel(
        saved_model_schema_version=1)
    with self.assertRaisesRegex(ValueError,
                                "Given proto could not be deserialized as"):
      fingerprinting.Fingerprint.from_proto(wrong_def)

  def test_fingerprint_to_proto(self):
    # [FUNCTION DOC]
    # Scenario: save a model, read its Fingerprint and convert it to a message with the
    # fingerprinting utilities. Expected: the Fingerprint equals the converted message.
    save_dir = os.path.join(self.get_temp_dir(), "from_proto_model")
    save.save(self._create_model_with_data(), save_dir)
    fingerprint = fingerprinting.read_fingerprint(save_dir)
    fingerprint_def = fingerprinting_utils.to_proto(fingerprint)
    self.assertEqual(fingerprint, fingerprint_def)


if __name__ == "__main__":
  test.main()
