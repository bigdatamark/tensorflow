# [MODULE DOC]
# Import router for the file I/O layer. Re-exports functions of the low-level file I/O module
# under the legacy capitalized gfile names (copy, make directory, remove, delete recursively,
# exists, glob, is-directory, list directory, make directories, rename, stat, walk) and defines
# two public file classes, the standard file wrapper and a deprecated v1-only fast variant,
# plus an alias named Open for the standard wrapper. Import-time side effects: registers the
# public API names through the export decorator. Not a helper module in itself, but mostly glue.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Import router for file_io."""
# pylint: disable=unused-import
from tensorflow.python.lib.io.file_io import copy as Copy
from tensorflow.python.lib.io.file_io import create_dir as MkDir
from tensorflow.python.lib.io.file_io import delete_file as Remove
from tensorflow.python.lib.io.file_io import delete_recursively as DeleteRecursively
from tensorflow.python.lib.io.file_io import file_exists as Exists
from tensorflow.python.lib.io.file_io import FileIO as _FileIO
from tensorflow.python.lib.io.file_io import get_matching_files as Glob
from tensorflow.python.lib.io.file_io import is_directory as IsDirectory
from tensorflow.python.lib.io.file_io import list_directory as ListDirectory
from tensorflow.python.lib.io.file_io import recursive_create_dir as MakeDirs
from tensorflow.python.lib.io.file_io import rename as Rename
from tensorflow.python.lib.io.file_io import stat as Stat
from tensorflow.python.lib.io.file_io import walk as Walk
# pylint: enable=unused-import
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


@tf_export('io.gfile.GFile', v1=['gfile.GFile', 'gfile.Open', 'io.gfile.GFile'])
class GFile(_FileIO):
  # [CLASS DOC]
  # Public file object backed by TensorFlow's file system layer, so paths may use local files or
  # remote URI schemes. Behavior is entirely inherited from the low-level file class; this class
  # only fixes the constructor signature (name and mode). Exported as the public gfile class
  # (also reachable under the legacy open and gfile names in v1). Usable as a context manager.
  r"""File I/O wrappers without thread locking.

  The main roles of the `tf.io.gfile` module are:

  1. To provide an API that is close to Python's file I/O objects, and
  2. To provide an implementation based on TensorFlow's C++ FileSystem API.

  The C++ FileSystem API supports multiple file system implementations,
  including local files, Google Cloud Storage (using a `gs://` prefix, and
  HDFS (using an `hdfs://` prefix). TensorFlow exports these as `tf.io.gfile`,
  so that you can use these implementations for saving and loading checkpoints,
  writing to TensorBoard logs, and accessing training data (among other uses).
  However, if all your files are local, you can use the regular Python file
  API without any problem.

  *Note*: though similar to Python's I/O implementation, there are semantic
  differences to make `tf.io.gfile` more efficient for backing filesystems. For
  example, a write mode file will not be opened until the first write call to
  minimize RPC invocations in network filesystems.

  Once you obtain a `GFile` object, you can use it in most ways as you would any
  Python's file object:

  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  4
  >>> with tf.io.gfile.GFile("/tmp/x") as f:
  ...   f.read()
  'asdf'

  The difference is that you can specify URI schemes to use other filesystems
  (e.g., `gs://` for GCS, `s3://` for S3, etc.), if they are supported. Using
  `file://` as an example, we have:

  >>> with tf.io.gfile.GFile("file:///tmp/x", "w") as f:
  ...   f.write("qwert")
  ...   f.write("asdf")
  >>> tf.io.gfile.GFile("file:///tmp/x").read()
  'qwertasdf'

  You can also read all lines of a file directly:

  >>> with tf.io.gfile.GFile("file:///tmp/x", "w") as f:
  ...   f.write("asdf\n")
  ...   f.write("qwer\n")
  >>> tf.io.gfile.GFile("/tmp/x").readlines()
  ['asdf\n', 'qwer\n']

  You can iterate over the lines:

  >>> with tf.io.gfile.GFile("file:///tmp/x", "w") as f:
  ...   f.write("asdf\n")
  ...   f.write("qwer\n")
  >>> for line in tf.io.gfile.GFile("/tmp/x"):
  ...   print(line[:-1]) # removes the end of line character
  asdf
  qwer

  Random access read is possible if the underlying filesystem supports it:

  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdfqwer")
  >>> f = tf.io.gfile.GFile("/tmp/x")
  >>> f.read(3)
  'asd'
  >>> f.seek(4)
  >>> f.tell()
  4
  >>> f.read(3)
  'qwe'
  >>> f.tell()
  7
  >>> f.close()
  """

  def __init__(self, name, mode='r'):
    # [FUNCTION DOC]
    # Constructor. Inputs: name is the file path or URI (string or path-like), mode is the access
    # mode string, defaulting to read. Delegates to the parent constructor; the file is opened
    # lazily by the parent on first use. Returns nothing.
    super(GFile, self).__init__(name=name, mode=mode)


@tf_export(v1=['gfile.FastGFile'])
class FastGFile(_FileIO):
  # [CLASS DOC]
  # Deprecated v1-only public file class, functionally identical to the standard gfile class
  # (inherits all behavior from the low-level file class). Its constructor is marked deprecated
  # and emits a warning pointing users to the standard class.
  """File I/O wrappers without thread locking.

  Note, that this  is somewhat like builtin Python  file I/O, but
  there are  semantic differences to  make it more  efficient for
  some backing filesystems.  For example, a write  mode file will
  not  be opened  until the  first  write call  (to minimize  RPC
  invocations in network filesystems).
  """

  @deprecated(None, 'Use tf.gfile.GFile.')
  def __init__(self, name, mode='r'):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring claims this class is faster or has distinct semantics, but
    # it adds nothing over the standard class; only the deprecation decorator differs.
    # Constructor, wrapped by a deprecation decorator that logs a warning on use. Inputs: name
    # (path or URI) and mode (string, default read). Delegates to the parent constructor.
    super(FastGFile, self).__init__(name=name, mode=mode)


# Does not alias to Open so that we use our version of GFile to strip
# 'b' mode.
Open = GFile
