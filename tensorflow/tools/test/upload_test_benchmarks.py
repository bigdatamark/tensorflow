# [MODULE DOC]
# Command-line script that uploads benchmark test result files to a Google Cloud Datastore.
# Scans a data directory for result files (JSON form of the TestResults message), converts each
# into one Test entity plus one Entry entity per benchmark entry, uploads them in a batch, and
# moves the processed file into an archive directory. Uses non-blocking exclusive file locks so
# several instances can run at once. Requires the credentials environment variable for the
# datastore. Exposes small file and lock helpers plus the upload and option-parsing functions.
# No import-time side effects other than importing the datastore client library.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Command to upload benchmark test results to a cloud datastore.

This uploader script is typically run periodically as a cron job.  It locates,
in a specified data directory, files that contain benchmark test results.  The
results are written by the "run_and_gather_logs.py" script using the JSON-format
serialization of the "TestResults" protobuf message (core/util/test_log.proto).

For each file, the uploader reads the "TestResults" data, transforms it into
the schema used in the datastore (see below), and upload it to the datastore.
After processing a file, the uploader moves it to a specified archive directory
for safe-keeping.

The uploader uses file-level exclusive locking (non-blocking flock) which allows
multiple instances of this script to run concurrently if desired, splitting the
task among them, each one processing and archiving different files.

The "TestResults" object contains test metadata and multiple benchmark entries.
The datastore schema splits this information into two Kinds (like tables), one
holding the test metadata in a single "Test" Entity (like rows), and one holding
each related benchmark entry in a separate "Entry" Entity.  Datastore create a
unique ID (retrieval key) for each Entity, and this ID is always returned along
with the data when an Entity is fetched.

* Test:
  - test:   unique name of this test (string)
  - start:  start time of this test run (datetime)
  - info:   JSON-encoded test metadata (string, not indexed)

* Entry:
  - test:   unique name of this test (string)
  - entry:  unique name of this benchmark entry within this test (string)
  - start:  start time of this test run (datetime)
  - timing: average time (usec) per iteration of this test/entry run (float)
  - info:   JSON-encoded entry metadata (string, not indexed)

A few composite indexes are created (upload_test_benchmarks_index.yaml) for fast
retrieval of benchmark data and reduced I/O to the client without adding a lot
of indexing and storage burden:

* Test: (test, start) is indexed to fetch recent start times for a given test.

* Entry: (test, entry, start, timing) is indexed to use projection and only
fetch the recent (start, timing) data for a given test/entry benchmark.

Example retrieval GQL statements:

* Get the recent start times for a given test:
  SELECT start FROM Test WHERE test = <test-name> AND
    start >= <recent-datetime> LIMIT <count>

* Get the recent timings for a given benchmark:
  SELECT start, timing FROM Entry WHERE test = <test-name> AND
    entry = <entry-name> AND start >= <recent-datetime> LIMIT <count>

* Get all test names uniquified (e.g. display a list of available tests):
  SELECT DISTINCT ON (test) test FROM Test

* For a given test (from the list above), get all its entry names.  The list of
  entry names can be extracted from the test "info" metadata for a given test
  name and start time (e.g. pick the latest start time for that test).
  SELECT * FROM Test WHERE test = <test-name> AND start = <latest-datetime>
"""

import argparse
import datetime
import fcntl
import json
import os
import shutil

from google.cloud import datastore


def is_real_file(dirpath, fname):
  # [FUNCTION DOC]
  # HELPER: file filter used when listing benchmark files to upload.
  # Returns true when the named entry inside the directory is a regular file and not a symbolic
  # link. Inputs: directory path and entry name. No side effects.
  fpath = os.path.join(dirpath, fname)
  return os.path.isfile(fpath) and not os.path.islink(fpath)


def get_mtime(dirpath, fname):
  # [FUNCTION DOC]
  # HELPER: sort key provider for ordering benchmark files by age.
  # Returns the last modification time (seconds since the epoch) of the named file in the
  # directory. Raises an OS error if the file cannot be examined. No other side effects.
  fpath = os.path.join(dirpath, fname)
  return os.stat(fpath).st_mtime


def list_files_by_mtime(dirpath):
  # [FUNCTION DOC]
  # Lists the regular, non-link files directly inside the given directory and returns their
  # names sorted from oldest to newest modification time. Raises an OS error if the directory
  # cannot be listed. No side effects.
  """Return a list of files in the directory, sorted in increasing "mtime".

  Return a list of files in the given directory, sorted from older to newer file
  according to their modification times.  Only return actual files, skipping
  directories, symbolic links, pipes, etc.

  Args:
    dirpath: directory pathname

  Returns:
    A list of file names relative to the given directory path.
  """
  files = [f for f in os.listdir(dirpath) if is_real_file(dirpath, f)]
  return sorted(files, key=lambda f: get_mtime(dirpath, f))


# Note: The file locking code uses flock() instead of lockf() because benchmark
# files are only opened for reading (not writing) and we still want exclusive
# locks on them.  This imposes the limitation that the data directory must be
# local, not NFS-mounted.
def lock(fd):
  # [FUNCTION DOC]
  # HELPER: file locking utility for the uploader.
  # KNOWN ISSUE: this function is never called in this module; only the non-blocking variant
  # is used, so it is currently dead code.
  # Takes an open file object and blocks until an exclusive advisory lock is acquired on it.
  # Returns nothing. The lock is tied to the descriptor and is released when it is closed.
  fcntl.flock(fd, fcntl.LOCK_EX)


def unlock(fd):
  # [FUNCTION DOC]
  # HELPER: file locking utility for the uploader.
  # KNOWN ISSUE: never called in this module (release happens implicitly when the file closes).
  # Takes an open file object and releases the advisory lock held on it. Returns nothing.
  fcntl.flock(fd, fcntl.LOCK_UN)


def trylock(fd):
  # [FUNCTION DOC]
  # HELPER: file locking utility for the uploader.
  # Attempts to take an exclusive advisory lock on the open file object without blocking.
  # Returns true if the lock was acquired and false on any failure (for example when another
  # process already holds it). Never raises.
  try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return True
  except Exception:  # pylint: disable=broad-except
    return False


def upload_benchmark_data(client, data):
  # [FUNCTION DOC]
  # Parses one benchmark result (JSON text) and uploads it to the datastore through the client.
  # Inputs: client is a datastore client connection; data is the JSON-encoded results text.
  # Builds one Test entity (name, start time converted from the epoch value, and the whole raw
  # text as a non-indexed info field) and one Entry entity per benchmark entry (test name,
  # start time, entry name, wall time as timing, and the entry JSON as non-indexed info).
  # All entities are written in one batch call. Returns nothing. Raises on malformed JSON or
  # missing keys (name, start time, entries), or if the datastore write fails.
  """Parse benchmark data and use the client to upload it to the datastore.

  Parse the given benchmark data from the serialized JSON-format used to write
  the test results file.  Create the different datastore Entities from that data
  and upload them to the datastore in a batch using the client connection.

  Args:
    client: datastore client connection
    data: JSON-encoded benchmark data
  """
  test_result = json.loads(data)

  test_name = str(test_result["name"])
  start_time = datetime.datetime.utcfromtimestamp(
      float(test_result["startTime"]))
  batch = []

  # Create the Test Entity containing all the test information as a
  # non-indexed JSON blob.
  t_key = client.key("Test")
  t_val = datastore.Entity(t_key, exclude_from_indexes=["info"])
  t_val.update({"test": test_name, "start": start_time, "info": str(data)})
  batch.append(t_val)

  # Create one Entry Entity for each benchmark entry.  The wall-clock timing is
  # the attribute to be fetched and displayed.  The full entry information is
  # also stored as a non-indexed JSON blob.
  for ent in test_result["entries"].get("entry", []):
    ent_name = str(ent["name"])
    e_key = client.key("Entry")
    e_val = datastore.Entity(e_key, exclude_from_indexes=["info"])
    e_val.update({
        "test": test_name,
        "start": start_time,
        "entry": ent_name,
        "timing": ent["wallTime"],
        "info": str(json.dumps(ent))
    })
    batch.append(e_val)

  # Put the whole batch of Entities in the datastore.
  client.put_multi(batch)


def upload_benchmark_files(opts):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the file is locked for processing, but a failed lock is
  # silently skipped and any exception is only printed, so errors never change the exit status.
  # Input: opts is the parsed options object with the data directory and archive directory.
  # Creates a datastore client, then visits files in the data directory from oldest to newest.
  # For each file it opens it, tries a non-blocking lock, and if successful uploads its
  # contents and moves the file into the archive directory. Files locked elsewhere are skipped;
  # any per-file error prints a message and processing continues. Returns nothing.
  """Find benchmark files, process them, and upload their data to the datastore.

  Locate benchmark files in the data directory, process them, and upload their
  data to the datastore.  After processing each file, move it to the archive
  directory for safe-keeping.  Each file is locked for processing, which allows
  multiple uploader instances to run concurrently if needed, each one handling
  different benchmark files, skipping those already locked by another.

  Args:
    opts: command line options object

  Note: To use locking, the file is first opened, then its descriptor is used to
  lock and read it.  The lock is released when the file is closed.  Do not open
  that same file a 2nd time while the lock is already held, because when that
  2nd file descriptor is closed, the lock will be released prematurely.
  """
  client = datastore.Client()

  for fname in list_files_by_mtime(opts.datadir):
    fpath = os.path.join(opts.datadir, fname)
    try:
      with open(fpath, "r") as fd:
        if trylock(fd):
          upload_benchmark_data(client, fd.read())
          shutil.move(fpath, os.path.join(opts.archivedir, fname))
          # unlock(fd) -- When "with open()" closes fd, the lock is released.
    except Exception as e:  # pylint: disable=broad-except
      print("Cannot process '%s', skipping. Error: %s" % (fpath, e))


def parse_cmd_line():
  # [FUNCTION DOC]
  # HELPER: command line parsing for the script entry point.
  # Defines two required string options, the archive directory and the data directory, parses
  # the process command line and returns the parsed arguments object. Exits with a usage
  # message if a required option is missing.
  """Parse command line options.

  Returns:
    The parsed arguments object.
  """
  desc = "Upload benchmark results to datastore."
  opts = [
      ("-a", "--archivedir", str, None, True,
       "Directory where benchmark files are archived."),
      ("-d", "--datadir", str, None, True,
       "Directory of benchmark files to upload."),
  ]

  parser = argparse.ArgumentParser(description=desc)
  for opt in opts:
    parser.add_argument(opt[0], opt[1], type=opt[2], default=opt[3],
                        required=opt[4], help=opt[5])
  return parser.parse_args()


def main():
  # [FUNCTION DOC]
  # Script entry point. Parses the command line, verifies that the datastore credentials
  # environment variable is set (raises a value error otherwise), then uploads all benchmark
  # files. Takes no inputs and returns nothing.
  options = parse_cmd_line()

  # Check that credentials are specified to access the datastore.
  if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS env. var. is not set.")

  upload_benchmark_files(options)


if __name__ == "__main__":
  main()
