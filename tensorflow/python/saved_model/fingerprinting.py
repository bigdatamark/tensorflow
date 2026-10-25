# [MODULE DOC]
# Python API for reading SavedModel fingerprints. Defines the Fingerprint value class
# (exported as the experimental saved model fingerprint type) and the read_fingerprint
# function (also exported), both backed by the native fingerprinting binding module and the
# fingerprint protocol buffer. Import-time side effect: registration of the public API exports.
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
"""Methods for SavedModel fingerprinting.

This module contains classes and functions for reading the SavedModel
fingerprint.
"""

from typing import Any

from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.util.tf_export import tf_export


@tf_export("saved_model.experimental.Fingerprint", v1=[])
class Fingerprint:
  # [CLASS DOC]
  # Plain value object holding the fields of a SavedModel fingerprint: saved model checksum,
  # graph definition program hash, signature definition hash, saved object graph hash,
  # checkpoint hash and the producer version. Created directly, from a fingerprint proto via
  # from_proto, or returned by read_fingerprint. Supports equality, string and representation
  # output, and computing a canonical singleprint identifier.
  # KNOWN ISSUE: equality and the string forms ignore the version field, and defining equality
  # without a hash method makes instances unhashable.
  """The SavedModel fingerprint.

  Each attribute of this class is named after a field name in the
  FingerprintDef proto and contains the value of the respective field in the
  protobuf.

  Attributes:
    saved_model_checksum: A uint64 containing the `saved_model_checksum`.
    graph_def_program_hash: A uint64 containing `graph_def_program_hash`.
    signature_def_hash: A uint64 containing the `signature_def_hash`.
    saved_object_graph_hash: A uint64 containing the `saved_object_graph_hash`.
    checkpoint_hash: A uint64 containing the`checkpoint_hash`.
    version: An int32 containing the producer field of the VersionDef.
  """

  def __init__(
      self,
      saved_model_checksum: int = None,
      graph_def_program_hash: int = None,
      signature_def_hash: int = None,
      saved_object_graph_hash: int = None,
      checkpoint_hash: int = None,
      version: int = None,
  ):
    # [FUNCTION DOC]
    # Constructor storing the six fingerprint values as attributes of the same names: saved model
    # checksum, graph definition program hash, signature definition hash, saved object graph
    # hash, checkpoint hash (all unsigned 64-bit style integers) and version (producer int).
    # Every parameter is optional and defaults to the null value. No validation is done and
    # nothing is returned.
    """Initializes the instance based on values in the SavedModel fingerprint.

    Args:
      saved_model_checksum: Value of the`saved_model_checksum`.
      graph_def_program_hash: Value of the `graph_def_program_hash`.
      signature_def_hash: Value of the `signature_def_hash`.
      saved_object_graph_hash: Value of the `saved_object_graph_hash`.
      checkpoint_hash: Value of the `checkpoint_hash`.
      version: Value of the producer field of the VersionDef.
    """
    self.saved_model_checksum = saved_model_checksum
    self.graph_def_program_hash = graph_def_program_hash
    self.signature_def_hash = signature_def_hash
    self.saved_object_graph_hash = saved_object_graph_hash
    self.checkpoint_hash = checkpoint_hash
    self.version = version

  @classmethod
  def from_proto(cls, proto: fingerprint_pb2.FingerprintDef) -> "Fingerprint":
    # [FUNCTION DOC]
    # KNOWN ISSUE: it builds a Fingerprint explicitly instead of the class it was called on, so
    # subclasses get a base instance.
    # Class method building a Fingerprint from a fingerprint definition message. If the input is
    # raw bytes it is first parsed into the message type. Copies the six fields in order and
    # returns the new instance. If the object lacks the expected fields (attribute error), raises
    # a value error saying the proto could not be deserialized as a fingerprint, with the
    # original error text appended and the original exception suppressed.
    """Constructs Fingerprint object from protocol buffer message."""
    if isinstance(proto, bytes):
      proto = fingerprint_pb2.FingerprintDef.FromString(proto)
    try:
      return Fingerprint(
          proto.saved_model_checksum,
          proto.graph_def_program_hash,
          proto.signature_def_hash,
          proto.saved_object_graph_hash,
          proto.checkpoint_hash,
          proto.version)
    except AttributeError as e:
      raise ValueError(
          f"Given proto could not be deserialized as fingerprint."
          f"{e}") from None

  def __eq__(self, other: Any) -> bool:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the version field is not compared, so fingerprints differing only in version
    # are considered equal.
    # Equality against another Fingerprint or a fingerprint definition message. Compares the
    # saved model checksum, graph def program hash, signature def hash, saved object graph hash
    # and checkpoint hash. Returns true when all are equal, false when any differs, when the
    # other object is of another type, or when an attribute is missing. Never raises.
    if (isinstance(other, Fingerprint) or
        isinstance(other, fingerprint_pb2.FingerprintDef)):
      try:
        return (
            self.saved_model_checksum == other.saved_model_checksum and
            self.graph_def_program_hash == other.graph_def_program_hash and
            self.signature_def_hash == other.signature_def_hash and
            self.saved_object_graph_hash == other.saved_object_graph_hash and
            self.checkpoint_hash == other.checkpoint_hash)
      except AttributeError:
        pass
    return False

  def __str__(self) -> str:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the version field is omitted from the output.
    # Returns a multi-line human readable string: a title line followed by one indented line for
    # each of the five hash fields (checksum, graph def program hash, signature def hash, saved
    # object graph hash, checkpoint hash).
    return "\n".join([
        "SavedModel Fingerprint",
        f"  saved_model_checksum: {self.saved_model_checksum}",
        f"  graph_def_program_hash: {self.graph_def_program_hash}",
        f"  signature_def_hash: {self.signature_def_hash}",
        f"  saved_object_graph_hash: {self.saved_object_graph_hash}",
        f"  checkpoint_hash: {self.checkpoint_hash}"
    ])

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the version field is omitted, so the representation does not contain all
    # constructor arguments.
    # Returns a constructor-call style string with the five hash fields in order.
    return (f"Fingerprint({self.saved_model_checksum}, "
            f"{self.graph_def_program_hash}, "
            f"{self.signature_def_hash}, "
            f"{self.saved_object_graph_hash}, "
            f"{self.checkpoint_hash})")

  def singleprint(self) -> fingerprinting_pywrap.Singleprint:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns the string concatenation separated by slashes,
    # but the code returns the native Singleprint object built from the four hashes (not
    # necessarily a plain string); also the error message lacks separators between fields.
    # Builds the canonical identifier of the SavedModel from the graph def program hash,
    # signature def hash, saved object graph hash and checkpoint hash (checksum excluded as it
    # is not deterministic). Raises a value error, with the field values and the original error
    # text, if the native layer raises a type error or a fingerprint error for invalid values.
    """Canonical fingerprinting ID for a SavedModel.

    Uniquely identifies a SavedModel based on the regularized fingerprint
    attributes. (saved_model_checksum is sensitive to immaterial changes and
    thus non-deterministic.)

    Returns:
      The string concatenation of `graph_def_program_hash`,
      `signature_def_hash`, `saved_object_graph_hash`, and `checkpoint_hash`
      fingerprint attributes (separated by '/').

    Raises:
      ValueError: If the fingerprint fields cannot be used to construct the
      singleprint.
    """
    try:
      return fingerprinting_pywrap.Singleprint(self.graph_def_program_hash,
                                               self.signature_def_hash,
                                               self.saved_object_graph_hash,
                                               self.checkpoint_hash)
    except (TypeError, fingerprinting_pywrap.FingerprintException) as e:
      raise ValueError(
          f"Encounted invalid fingerprint values when constructing singleprint."
          f"graph_def_program_hash: {self.graph_def_program_hash}"
          f"signature_def_hash: {self.signature_def_hash}"
          f"saved_object_graph_hash: {self.saved_object_graph_hash}"
          f"checkpoint_hash: {self.checkpoint_hash}"
          f"{e}") from None


@tf_export("saved_model.experimental.read_fingerprint", v1=[])
def read_fingerprint(export_dir: str) -> Fingerprint:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a missing or invalid fingerprint raises file-not-found, but
  # only the native file-not-found condition does; other native fingerprint errors surface as a
  # runtime error.
  # Reads the fingerprint file stored in the given SavedModel export directory (string path)
  # through the native layer. Returns a Fingerprint built from the parsed message. Raises
  # file-not-found error when the native layer reports a missing file, and a runtime error for
  # any other native fingerprint failure. Reads from disk; no other side effects.
  """Reads the fingerprint of a SavedModel in `export_dir`.

  Returns a `tf.saved_model.experimental.Fingerprint` object that contains
  the values of the SavedModel fingerprint, which is persisted on disk in the
  `fingerprint.pb` file in the `export_dir`.

  Read more about fingerprints in the SavedModel guide at
  https://www.tensorflow.org/guide/saved_model.

  Args:
    export_dir: The directory that contains the SavedModel.

  Returns:
    A `tf.saved_model.experimental.Fingerprint`.

  Raises:
    FileNotFoundError: If no or an invalid fingerprint is found.
  """
  try:
    fingerprint = fingerprinting_pywrap.ReadSavedModelFingerprint(export_dir)
  except fingerprinting_pywrap.FileNotFoundException as e:
    raise FileNotFoundError(f"SavedModel Fingerprint Error: {e}") from None  # pylint: disable=raise-missing-from
  except fingerprinting_pywrap.FingerprintException as e:
    raise RuntimeError(f"SavedModel Fingerprint Error: {e}") from None  # pylint: disable=raise-missing-from
  return Fingerprint.from_proto(
      fingerprint_pb2.FingerprintDef().FromString(fingerprint))
