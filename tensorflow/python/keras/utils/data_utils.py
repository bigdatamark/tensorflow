# [MODULE DOC]
# Utilities for downloading and caching files (with hash validation and archive extraction) and
# for feeding Keras training from sequences and generators: the Sequence base class, a thread-safe
# generator wrapper, and background enqueuers using thread or process pools with module-level
# shared state (sequence map, id counter, pool registry, force-thread-pool flag). Import-time:
# defines that module state and a Python 2 download fallback that is never used on Python 3.
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
# pylint: disable=g-import-not-at-top
"""Utilities for file download and caching."""

from abc import abstractmethod
from contextlib import closing
import functools
import hashlib
import multiprocessing
import multiprocessing.dummy
import os
import queue
import random
import shutil
import sys  # pylint: disable=unused-import
import tarfile
import threading
import time
import typing
import urllib
import weakref
import zipfile

import numpy as np

from tensorflow.python.framework import tensor
from six.moves.urllib.request import urlopen
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string

# Required to support google internal urlretrieve
if sys.version_info[0] == 2:

  def urlretrieve(url, filename, reporthook=None, data=None):
    # [FUNCTION DOC]
    # HELPER: legacy replacement for the standard download function, defined only under Python 2
    # (dead code on modern Python, since the version check is false).
    # Opens the url (optionally with request data), streams the body in chunks and writes it to
    # the local file named filename, calling reporthook (optional callback taking a block count,
    # block size and total size, -1 when unknown) once per chunk read. Returns nothing. Side
    # effects: network access and creation or overwriting of a local file.
    """Replacement for `urlretrieve` for Python 2.

    Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
    `urllib` module, known to have issues with proxy management.

    Args:
        url: url to retrieve.
        filename: where to store the retrieved data locally.
        reporthook: a hook function that will be called once on establishment of
          the network connection and once after each block read thereafter. The
          hook will be passed three arguments; a count of blocks transferred so
          far, a block size in bytes, and the total size of the file.
        data: `data` argument passed to `urlopen`.
    """

    def chunk_read(response, chunk_size=8192, reporthook=None):
      # [FUNCTION DOC]
      # HELPER: generator for the Python 2 urlretrieve replacement.
      # Reads the response in chunks of chunk_size bytes (default 8192), determining total size
      # from the content-length header (-1 if absent). For every read, including the final empty
      # one, calls the optional reporthook with the running count, chunk size and total size.
      # Yields each non-empty chunk of bytes and stops at the first empty read.
      content_type = response.info().get('Content-Length')
      total_size = -1
      if content_type is not None:
        total_size = int(content_type.strip())
      count = 0
      while True:
        chunk = response.read(chunk_size)
        count += 1
        if reporthook is not None:
          reporthook(count, chunk_size, total_size)
        if chunk:
          yield chunk
        else:
          break

    response = urlopen(url, data)
    with open(filename, 'wb') as fd:
      for chunk in chunk_read(response, reporthook=reporthook):
        fd.write(chunk)
else:
  from urllib.request import urlretrieve  # pylint: disable=g-importing-member


def is_generator_or_sequence(x):
  # [FUNCTION DOC]
  # Decides whether x is a Keras-style data generator: a generator, a Sequence, or any iterator.
  # Tensors, arrays and built-in containers or strings are explicitly rejected. Input: x, any
  # object. Returns True or False. No side effects.
  """Check if `x` is a Keras generator type."""
  builtin_iterators = (str, list, tuple, dict, set, frozenset)
  if isinstance(x, (tensor.Tensor, np.ndarray) + builtin_iterators):
    return False
  return (tf_inspect.isgenerator(x) or
          isinstance(x, Sequence) or
          isinstance(x, typing.Iterator))


def _extract_archive(file_path, path='.', archive_format='auto'):
  # [FUNCTION DOC]
  # HELPER: archive extraction step of the file download function.
  # KNOWN ISSUE: extractall is used without validating member paths, so a malicious archive can
  # write outside the target (path traversal). Also, on failure the whole target path is removed,
  # which can delete a pre-existing directory.
  # Inputs: file_path, archive path; path, destination (default current directory); archive_format,
  # one of auto (tar then zip), tar, zip, a list of those, or nothing (no extraction). Tries each
  # format whose detector matches the file and extracts into path. Returns True if an archive was
  # extracted, else False. On a tar error, runtime error or keyboard interrupt, deletes the target
  # path (file or directory) and re-raises.
  """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

  Args:
      file_path: path to the archive file
      path: path to extract the archive file
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.

  Returns:
      True if a match was found and an archive extraction was completed,
      False otherwise.
  """
  if archive_format is None:
    return False
  if archive_format == 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, str):
    archive_format = [archive_format]

  file_path = path_to_string(file_path)
  path = path_to_string(path)

  for archive_type in archive_format:
    if archive_type == 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type == 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          archive.extractall(path)
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          if os.path.exists(path):
            if os.path.isfile(path):
              os.remove(path)
            else:
              shutil.rmtree(path)
          raise
      return True
  return False


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: when untar is true the file is saved with a tar.gz suffix and extraction is
  # forced to tar, ignoring archive_format and extract; the docstring omits that. Also the
  # download progress bar and hash message use plain printing, and if the cache directory is not
  # writable it silently falls back to a temp directory.
  # Downloads a file into a local cache unless it is already present and valid. Inputs: fname
  # (name or absolute path), origin (url), untar (deprecated, extract as tar), md5_hash
  # (deprecated, used as file_hash with md5 if file_hash is absent), file_hash, cache_subdir,
  # hash_algorithm (auto, md5 or sha256), extract, archive_format, cache_dir (default user home
  # keras folder). Creates directories, downloads with a progress bar, deletes a partial file on
  # failure or interrupt, and re-downloads if the hash mismatches. Returns the path of the file, or
  # of the extracted target when untar is true. Raises a generic exception on network failures.
  """Downloads a file from a URL if it not already in the cache.

  By default the file at the url `origin` is downloaded to the
  cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
  and given the filename `fname`. The final location of a file
  `example.txt` would therefore be `~/.keras/datasets/example.txt`.

  Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
  Passing a hash will verify the file after download. The command line
  programs `shasum` and `sha256sum` can compute the hash.

  Example:

  ```python
  path_to_downloaded_file = tf.keras.utils.get_file(
      "flower_photos",
      "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
      untar=True)
  ```

  Args:
      fname: Name of the file. If an absolute path `/path/to/file.txt` is
          specified the file will be saved at that location.
      origin: Original URL of the file.
      untar: Deprecated in favor of `extract` argument.
          boolean, whether the file should be decompressed
      md5_hash: Deprecated in favor of `file_hash` argument.
          md5 hash of the file for verification
      file_hash: The expected hash string of the file after download.
          The sha256 and md5 hash algorithms are both supported.
      cache_subdir: Subdirectory under the Keras cache dir where the file is
          saved. If an absolute path `/path/to/folder` is
          specified the file will be saved at that location.
      hash_algorithm: Select the hash algorithm to verify the file.
          options are `'md5'`, `'sha256'`, and `'auto'`.
          The default 'auto' detects the hash algorithm in use.
      extract: True tries extracting the file as an Archive, like tar or zip.
      archive_format: Archive format to try for extracting the file.
          Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
          `'tar'` includes tar, tar.gz, and tar.bz files.
          The default `'auto'` corresponds to `['tar', 'zip']`.
          None or an empty list will return no matches found.
      cache_dir: Location to store cached files, when None it
          defaults to the default directory `~/.keras/`.

  Returns:
      Path to the downloaded file
  """
  if cache_dir is None:
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
  if md5_hash is not None and file_hash is None:
    file_hash = md5_hash
    hash_algorithm = 'md5'
  datadir_base = os.path.expanduser(cache_dir)
  if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
  datadir = os.path.join(datadir_base, cache_subdir)
  _makedirs_exist_ok(datadir)

  fname = path_to_string(fname)

  if untar:
    untar_fpath = os.path.join(datadir, fname)
    fpath = untar_fpath + '.tar.gz'
  else:
    fpath = os.path.join(datadir, fname)

  download = False
  if os.path.exists(fpath):
    # File found; verify integrity if a hash was provided.
    if file_hash is not None:
      if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
        print('A local file was found, but it seems to be '
              'incomplete or outdated because the ' + hash_algorithm +
              ' file hash does not match the original value of ' + file_hash +
              ' so we will re-download the data.')
        download = True
  else:
    download = True

  if download:
    print('Downloading data from', origin)

    class ProgressTracker(object):
      # [CLASS DOC]
      # HELPER: holder of the download progress bar for the lifetime of one download. Used as a
      # simple namespace; its only state is a class-level progress bar, initially nothing,
      # created lazily by the progress callback and reset after download.
      # Maintain progbar for the lifetime of download.
      # This design was chosen for Python 2.7 compatibility.
      progbar = None

    def dl_progress(count, block_size, total_size):
      # [FUNCTION DOC]
      # HELPER: progress callback passed to the download function.
      # KNOWN ISSUE: the first call only creates the progress bar and does not update it with the
      # first block's progress.
      # Inputs: count, number of blocks transferred; block_size, bytes per block; total_size,
      # total bytes or -1 if unknown. Creates the shared progress bar on first call (unknown total
      # becomes nothing) and later updates it with bytes transferred. Returns nothing.
      if ProgressTracker.progbar is None:
        if total_size == -1:
          total_size = None
        ProgressTracker.progbar = Progbar(total_size)
      else:
        ProgressTracker.progbar.update(count * block_size)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
      try:
        urlretrieve(origin, fpath, dl_progress)
      except urllib.error.HTTPError as e:
        raise Exception(error_msg.format(origin, e.code, e.msg))
      except urllib.error.URLError as e:
        raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise
    ProgressTracker.progbar = None

  if untar:
    if not os.path.exists(untar_fpath):
      _extract_archive(fpath, datadir, archive_format='tar')
    return untar_fpath

  if extract:
    _extract_archive(fpath, datadir, archive_format)

  return fpath


def _makedirs_exist_ok(datadir):
  # [FUNCTION DOC]
  # HELPER: creates the directory (and parents) if missing, tolerating existing ones.
  # Input: datadir, a directory path. Returns nothing. Side effect: filesystem directory creation.
  os.makedirs(datadir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg


def _resolve_hasher(algorithm, file_hash=None):
  # [FUNCTION DOC]
  # HELPER: chooses a hash object for file validation.
  # Returns a new sha256 hasher when algorithm is sha256, or when it is auto and file_hash is a
  # 64-character string; otherwise an md5 hasher (legacy default, also for any unknown name).
  # Inputs: algorithm name; file_hash, optional expected hash used to detect the algorithm.
  """Returns hash algorithm as hashlib function."""
  if algorithm == 'sha256':
    return hashlib.sha256()

  if algorithm == 'auto' and file_hash is not None and len(file_hash) == 64:
    return hashlib.sha256()

  # This is used only for legacy purposes.
  return hashlib.md5()


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
  # [FUNCTION DOC]
  # HELPER: computes a file digest for validation.
  # KNOWN ISSUE: a string algorithm is resolved without a file hash, so auto yields md5 here.
  # Inputs: fpath, file path; algorithm, a name string or an existing hasher object (which is
  # updated in place); chunk_size, bytes read per step. Reads the file in binary chunks. Returns
  # the hexadecimal digest string. Raises file errors if unreadable.
  """Calculates a file sha256 or md5 hash.

  Example:

  ```python
  _hash_file('/path/to/file.zip')
  'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
  ```

  Args:
      fpath: path to the file being validated
      algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
          The default `'auto'` detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.

  Returns:
      The file hash
  """
  if isinstance(algorithm, str):
    hasher = _resolve_hasher(algorithm)
  else:
    hasher = algorithm

  with open(fpath, 'rb') as fpath_file:
    for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
      hasher.update(chunk)

  return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
  # [FUNCTION DOC]
  # Checks that a file's digest equals the expected hash. Inputs: fpath; file_hash, expected
  # hexadecimal string; algorithm, auto (detected from hash length), sha256 or md5; chunk_size.
  # Returns True if the digests match (compared as strings), else False. Reads the whole file.
  """Validates a file against a sha256 or md5 hash.

  Args:
      fpath: path to the file being validated
      file_hash:  The expected hash string of the file.
          The sha256 and md5 hash algorithms are both supported.
      algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
          The default 'auto' detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.

  Returns:
      Whether the file is valid
  """
  hasher = _resolve_hasher(algorithm, file_hash)

  if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
    return True
  else:
    return False


class ThreadsafeIter(object):
  # [CLASS DOC]
  # HELPER: wraps an iterator so multiple threads can call next safely. Holds the wrapped
  # iterator, a lock, and the first exception raised, which is re-raised on all later calls.
  # Created by the thread-safe generator decorator.
  """Wrap an iterator with a lock and propagate exceptions to all threads."""

  def __init__(self, it):
    # [FUNCTION DOC]
    # Constructor. Input: it, the iterator to wrap. Stores it, creates a lock and clears the
    # remembered exception. No other side effects.
    self.it = it
    self.lock = threading.Lock()

    # After a generator throws an exception all subsequent next() calls raise a
    # StopIteration Exception. This, however, presents an issue when mixing
    # generators and threading because it means the order of retrieval need not
    # match the order in which the generator was called. This can make it appear
    # that a generator exited normally when in fact the terminating exception is
    # just in a different thread. In order to provide thread safety, once
    # self.it has thrown an exception we continue to throw the same exception.
    self._exception = None

  def __iter__(self):
    # [FUNCTION DOC]
    # Iterator protocol: returns the wrapper itself.
    return self

  def next(self):
    # [FUNCTION DOC]
    # HELPER: Python 2 style alias that simply delegates to the standard next-item method.
    return self.__next__()

  def __next__(self):
    # [FUNCTION DOC]
    # Returns the next item of the wrapped iterator while holding the lock. If an earlier call
    # raised, re-raises that same stored exception. Any exception from the wrapped iterator
    # (including end of iteration) is stored and re-raised. Side effect: updates stored exception.
    with self.lock:
      if self._exception:
        raise self._exception  # pylint: disable=raising-bad-type

      try:
        return next(self.it)
      except Exception as e:
        self._exception = e
        raise


def threadsafe_generator(f):
  # [FUNCTION DOC]
  # Decorator for generator functions. Input: f, a generator function. Returns a wrapper function
  # that calls f with any arguments and wraps the resulting generator in a thread-safe iterator.
  # The wrapper keeps f's metadata.

  @functools.wraps(f)
  def g(*a, **kw):
    # [FUNCTION DOC]
    # Wrapper produced by the decorator: calls the original generator function with all given
    # arguments and returns its generator wrapped in a thread-safe iterator.
    return ThreadsafeIter(f(*a, **kw))

  return g


class Sequence(object):
  # [CLASS DOC]
  # KNOWN ISSUE: the class does not use an abstract base class metaclass, so the abstract-method
  # decorators on its methods are not enforced; subclasses missing them fail only at call time.
  # Base class for datasets that yield whole batches by index, safe for multiprocessing. Subclasses
  # implement item access (returns one batch) and length (number of batches), and may override the
  # end-of-epoch hook. Iterating yields every batch once in order. Holds no state of its own.
  """Base object for fitting to a sequence of data, such as a dataset.

  Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
  If you want to modify your dataset between epochs you may implement
  `on_epoch_end`.
  The method `__getitem__` should return a complete batch.

  Notes:

  `Sequence` are a safer way to do multiprocessing. This structure guarantees
  that the network will only train once
   on each sample per epoch which is not the case with generators.

  Examples:

  ```python
  from skimage.io import imread
  from skimage.transform import resize
  import numpy as np
  import math

  # Here, `x_set` is list of path to the images
  # and `y_set` are the associated classes.

  class CIFAR10Sequence(Sequence):

      def __init__(self, x_set, y_set, batch_size):
          self.x, self.y = x_set, y_set
          self.batch_size = batch_size

      def __len__(self):
          return math.ceil(len(self.x) / self.batch_size)

      def __getitem__(self, idx):
          batch_x = self.x[idx * self.batch_size:(idx + 1) *
          self.batch_size]
          batch_y = self.y[idx * self.batch_size:(idx + 1) *
          self.batch_size]

          return np.array([
              resize(imread(file_name), (200, 200))
                 for file_name in batch_x]), np.array(batch_y)
  ```
  """

  @abstractmethod
  def __getitem__(self, index):
    # [FUNCTION DOC]
    # Abstract: subclasses return the batch at position index. This base version raises a
    # not-implemented error.
    """Gets batch at position `index`.

    Args:
        index: position of the batch in the Sequence.

    Returns:
        A batch
    """
    raise NotImplementedError

  @abstractmethod
  def __len__(self):
    # [FUNCTION DOC]
    # Abstract: subclasses return the number of batches. This base version raises a
    # not-implemented error.
    """Number of batch in the Sequence.

    Returns:
        The number of batches in the Sequence.
    """
    raise NotImplementedError

  def on_epoch_end(self):
    # [FUNCTION DOC]
    # Optional hook called after each epoch (for example to reshuffle data). Default does nothing
    # and returns nothing.
    """Method called at the end of every epoch.
    """
    pass

  def __iter__(self):
    # [FUNCTION DOC]
    # Generator yielding every batch of the sequence once, in index order, by indexing from zero
    # up to the length. Raises whatever item access raises.
    """Create a generator that iterate over the Sequence."""
    for item in (self[i] for i in range(len(self))):
      yield item


def iter_sequence_infinite(seq):
  # [FUNCTION DOC]
  # HELPER: endless iteration over a Sequence for training loops. Input: seq, a Sequence. Yields
  # its batches in order, restarting from the beginning after the last one, forever. Does not
  # call the end-of-epoch hook.
  """Iterates indefinitely over a Sequence.

  Args:
    seq: `Sequence` instance.

  Yields:
    Batches of data from the `Sequence`.
  """
  while True:
    for item in seq:
      yield item


# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None


# Because multiprocessing pools are inherently unsafe, starting from a clean
# state can be essential to avoiding deadlocks. In order to accomplish this, we
# need to be able to check on the status of Pools that we create.
_DATA_POOLS = weakref.WeakSet()
_WORKER_ID_QUEUE = None  # Only created if needed.
_WORKER_IDS = set()
_FORCE_THREADPOOL = False
_FORCE_THREADPOOL_LOCK = threading.RLock()


def dont_use_multiprocessing_pool(f):
  # [FUNCTION DOC]
  # HELPER: decorator that forces thread pools instead of process pools while f runs. Input: f,
  # a callable. Returns a wrapper with f's metadata.
  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the global flag is restored without a try-finally, so if the wrapped callable
    # raises the flag stays set and all later pools become thread pools.
    # Under a global lock, sets the module-wide force-thread-pool flag, calls the wrapped function
    # with the given arguments, restores the previous flag value and returns the result.
    with _FORCE_THREADPOOL_LOCK:
      global _FORCE_THREADPOOL
      old_force_threadpool, _FORCE_THREADPOOL = _FORCE_THREADPOOL, True
      out = f(*args, **kwargs)
      _FORCE_THREADPOOL = old_force_threadpool
      return out
  return wrapped


def get_pool_class(use_multiprocessing):
  # [FUNCTION DOC]
  # HELPER: selects the worker pool implementation. Input: use_multiprocessing, boolean. Returns
  # the thread pool class if multiprocessing is off or the module-wide force-thread-pool flag is
  # set; otherwise the process pool class. Reads module state only.
  global _FORCE_THREADPOOL
  if not use_multiprocessing or _FORCE_THREADPOOL:
    return multiprocessing.dummy.Pool  # ThreadPool
  return multiprocessing.Pool


def get_worker_id_queue():
  # [FUNCTION DOC]
  # HELPER: returns the shared multiprocessing queue used to record worker process ids, creating
  # it on first use. Side effect: sets the module-level queue. Not guarded by a lock.
  """Lazily create the queue to track worker ids."""
  global _WORKER_ID_QUEUE
  if _WORKER_ID_QUEUE is None:
    _WORKER_ID_QUEUE = multiprocessing.Queue()
  return _WORKER_ID_QUEUE


def init_pool(seqs):
  # [FUNCTION DOC]
  # HELPER: pool initializer. Input: seqs, mapping from identifiers to sequences. Replaces the
  # module-level shared mapping of sequences in the worker. Returns nothing.
  global _SHARED_SEQUENCES
  _SHARED_SEQUENCES = seqs


def get_index(uid, i):
  # [FUNCTION DOC]
  # HELPER: worker task for the ordered enqueuer. Inputs: uid, integer identifying a shared
  # sequence; i, batch index. Returns the batch at index i of that sequence from the shared module
  # mapping. Raises a key error if the identifier is unknown, or whatever item access raises.
  """Get the value from the Sequence `uid` at index `i`.

  To allow multiple Sequences to be used at the same time, we use `uid` to
  get a specific one. A single Sequence would cause the validation to
  overwrite the training Sequence.

  Args:
      uid: int, Sequence identifier
      i: index

  Returns:
      The value at index `i`.
  """
  return _SHARED_SEQUENCES[uid][i]


class SequenceEnqueuer(object):
  # [CLASS DOC]
  # Abstract base for background batch producers using worker threads or processes. State: the
  # sequence source, a multiprocessing flag, a unique id (from a module-wide counter), worker
  # count, executor factory, bounded queue, a run thread and a stop event. Subclasses implement the
  # run loop, pool initializer and the consumer generator. Start launches a daemon thread; stop
  # signals it and clears the queue. Writes the shared sequence mapping.
  """Base class to enqueue inputs.

  The task of an Enqueuer is to use parallelism to speed up preprocessing.
  This is done with processes or threads.

  Example:

  ```python
      enqueuer = SequenceEnqueuer(...)
      enqueuer.start()
      datas = enqueuer.get()
      for data in datas:
          # Use the inputs; training, evaluating, predicting.
          # ... stop sometime.
      enqueuer.stop()
  ```

  The `enqueuer.get()` should be an infinite stream of datas.
  """

  def __init__(self, sequence,
               use_multiprocessing=False):
    # [FUNCTION DOC]
    # Constructor. Inputs: sequence, the data source; use_multiprocessing, boolean (default false).
    # Allocates a unique id from the module-wide counter (a shared integer, falling back to a plain
    # integer if the OS disallows shared memory), updating the counter. Initializes worker count,
    # executor factory, queue, thread and stop event to empty values.
    self.sequence = sequence
    self.use_multiprocessing = use_multiprocessing

    global _SEQUENCE_COUNTER
    if _SEQUENCE_COUNTER is None:
      try:
        _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
      except OSError:
        # In this case the OS does not allow us to use
        # multiprocessing. We resort to an int
        # for enqueuer indexing.
        _SEQUENCE_COUNTER = 0

    if isinstance(_SEQUENCE_COUNTER, int):
      self.uid = _SEQUENCE_COUNTER
      _SEQUENCE_COUNTER += 1
    else:
      # Doing Multiprocessing.Value += x is not process-safe.
      with _SEQUENCE_COUNTER.get_lock():
        self.uid = _SEQUENCE_COUNTER.value
        _SEQUENCE_COUNTER.value += 1

    self.workers = 0
    self.executor_fn = None
    self.queue = None
    self.run_thread = None
    self.stop_signal = None

  def is_running(self):
    # [FUNCTION DOC]
    # Returns True if start was called and the stop event has not been set, else False.
    return self.stop_signal is not None and not self.stop_signal.is_set()

  def start(self, workers=1, max_queue_size=10):
    # [FUNCTION DOC]
    # Starts background production. Inputs: workers, number of workers (default 1); max_queue_size,
    # capacity of the result queue (default 10). Chooses the executor factory (process pool
    # initializer, or a thread pool factory), creates the queue and stop event, and launches a
    # daemon thread running the subclass run loop. Returns nothing.
    """Starts the handler's workers.

    Args:
        workers: Number of workers.
        max_queue_size: queue size
            (when full, workers could block on `put()`)
    """
    if self.use_multiprocessing:
      self.executor_fn = self._get_executor_init(workers)
    else:
      # We do not need the init since it's threads.
      self.executor_fn = lambda _: get_pool_class(False)(workers)
    self.workers = workers
    self.queue = queue.Queue(max_queue_size)
    self.stop_signal = threading.Event()
    self.run_thread = threading.Thread(target=self._run)
    self.run_thread.daemon = True
    self.run_thread.start()

  def _send_sequence(self):
    # [FUNCTION DOC]
    # HELPER: publishes this enqueuer's sequence in the module-wide shared mapping under its id so
    # newly created worker processes can see it. Returns nothing.
    """Sends current Iterable to all workers."""
    # For new processes that may spawn
    _SHARED_SEQUENCES[self.uid] = self.sequence

  def stop(self, timeout=None):
    # [FUNCTION DOC]
    # Stops production. Input: timeout, optional maximum seconds to wait for the run thread.
    # Sets the stop event, empties the queue under its mutex (resetting unfinished task count and
    # waking blocked producers), joins the thread, and clears this enqueuer's entry in the shared
    # mapping. Fails if start was never called (stop event is nothing).
    """Stops running threads and wait for them to exit, if necessary.

    Should be called by the same thread which called `start()`.

    Args:
        timeout: maximum time to wait on `thread.join()`
    """
    self.stop_signal.set()
    with self.queue.mutex:
      self.queue.queue.clear()
      self.queue.unfinished_tasks = 0
      self.queue.not_full.notify()
    self.run_thread.join(timeout)
    _SHARED_SEQUENCES[self.uid] = None

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor: stops the enqueuer if it is still running. Returns nothing.
    if self.is_running():
      self.stop()

  @abstractmethod
  def _run(self):
    # [FUNCTION DOC]
    # HELPER: abstract run loop executed on the background thread; subclasses submit work to the
    # executor and put result futures on the queue. Base version raises not-implemented.
    """Submits request to the executor and queue the `Future` objects."""
    raise NotImplementedError

  @abstractmethod
  def _get_executor_init(self, workers):
    # [FUNCTION DOC]
    # HELPER: abstract; subclasses return a function that builds a worker pool initialized with
    # the shared sequences. Input: workers, number of workers. Base version raises not-implemented.
    """Gets the Pool initializer for multiprocessing.

    Args:
        workers: Number of workers.

    Returns:
        Function, a Function to initialize the pool
    """
    raise NotImplementedError

  @abstractmethod
  def get(self):
    # [FUNCTION DOC]
    # Abstract consumer generator; subclasses yield produced batches from the queue. Base version
    # raises not-implemented.
    """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.
    # Returns
        Generator yielding tuples `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
    """
    raise NotImplementedError


class OrderedEnqueuer(SequenceEnqueuer):
  # [CLASS DOC]
  # Enqueuer for a Sequence that produces batches in index order, epoch after epoch, optionally
  # reshuffling the index order each epoch. Calls the sequence's end-of-epoch hook between epochs.
  # Extra state: the shuffle flag.
  """Builds a Enqueuer from a Sequence.

  Args:
      sequence: A `tf.keras.utils.data_utils.Sequence` object.
      use_multiprocessing: use multiprocessing if True, otherwise threading
      shuffle: whether to shuffle the data at the beginning of each epoch
  """

  def __init__(self, sequence, use_multiprocessing=False, shuffle=False):
    # [FUNCTION DOC]
    # Constructor. Inputs: sequence, a Sequence; use_multiprocessing, boolean (default false);
    # shuffle, boolean (default false) to reshuffle batch order each epoch. Calls the base
    # constructor and stores the shuffle flag.
    super(OrderedEnqueuer, self).__init__(sequence, use_multiprocessing)
    self.shuffle = shuffle

  def _get_executor_init(self, workers):
    # [FUNCTION DOC]
    # HELPER: returns a factory that builds a worker pool (process pool unless forced to threads)
    # of the given size, initialized with the shared sequences, no random seed and the worker id
    # queue; each created pool is registered in the module-wide pool set. Input: workers.
    """Gets the Pool initializer for multiprocessing.

    Args:
        workers: Number of workers.

    Returns:
        Function, a Function to initialize the pool
    """
    def pool_fn(seqs):
      # [FUNCTION DOC]
      # HELPER: factory returned above. Input: seqs, shared sequence mapping. Creates and returns a
      # worker pool initialized with it; registers the pool in the module-wide pool set.
      pool = get_pool_class(True)(
          workers, initializer=init_pool_generator,
          initargs=(seqs, None, get_worker_id_queue()))
      _DATA_POOLS.add(pool)
      return pool

    return pool_fn

  def _wait_queue(self):
    # [FUNCTION DOC]
    # HELPER: blocks, polling every tenth of a second, until all queued work is marked done or the
    # stop event is set. Returns nothing.
    """Wait for the queue to be empty."""
    while True:
      time.sleep(0.1)
      if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
        return

  def _run(self):
    # [FUNCTION DOC]
    # HELPER: background loop. Per epoch: optionally shuffles the batch indices, creates a pool,
    # submits an asynchronous fetch for each index (blocking on the bounded queue), waits for the
    # queue to drain, then calls the sequence's end-of-epoch hook and republishes the sequence.
    # Returns when the stop event is set.
    """Submits request to the executor and queue the `Future` objects."""
    sequence = list(range(len(self.sequence)))
    self._send_sequence()  # Share the initial sequence
    while True:
      if self.shuffle:
        random.shuffle(sequence)

      with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
        for i in sequence:
          if self.stop_signal.is_set():
            return

          self.queue.put(
              executor.apply_async(get_index, (self.uid, i)), block=True)

        # Done with the current epoch, waiting for the final batches
        self._wait_queue()

        if self.stop_signal.is_set():
          # We're done
          return

      # Call the internal on epoch end.
      self.sequence.on_epoch_end()
      self._send_sequence()  # Update the pool

  def get(self):
    # [FUNCTION DOC]
    # Generator yielding produced batches while the enqueuer runs: takes a future from the queue
    # (5 second timeout, retrying on empty), obtains its result, marks the task done and yields it
    # unless it is nothing. On any other error it stops the enqueuer and re-raises.
    """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.

    Yields:
        The next element in the queue, i.e. a tuple
        `(inputs, targets)` or
        `(inputs, targets, sample_weights)`.
    """
    while self.is_running():
      try:
        inputs = self.queue.get(block=True, timeout=5).get()
        if self.is_running():
          self.queue.task_done()
        if inputs is not None:
          yield inputs
      except queue.Empty:
        pass
      except Exception as e:  # pylint: disable=broad-except
        self.stop()
        raise e


def init_pool_generator(gens, random_seed=None, id_queue=None):
  # [FUNCTION DOC]
  # HELPER: initializer run in each pool worker. Inputs: gens, shared sources mapping; random_seed,
  # optional base seed; id_queue, optional queue of worker ids. Replaces the module-level shared
  # mapping, renames the worker process with a Keras prefix, seeds numpy randomness with the seed
  # plus the process id if a seed is given, and records the process id on the queue (blocking up
  # to 0.1 seconds, may raise a full-queue error).
  """Initializer function for pool workers.

  Args:
    gens: State which should be made available to worker processes.
    random_seed: An optional value with which to seed child processes.
    id_queue: A multiprocessing Queue of worker ids. This is used to indicate
      that a worker process was created by Keras and can be terminated using
      the cleanup_all_keras_forkpools utility.
  """
  global _SHARED_SEQUENCES
  _SHARED_SEQUENCES = gens

  worker_proc = multiprocessing.current_process()

  # name isn't used for anything, but setting a more descriptive name is helpful
  # when diagnosing orphaned processes.
  worker_proc.name = 'Keras_worker_{}'.format(worker_proc.name)

  if random_seed is not None:
    np.random.seed(random_seed + worker_proc.ident)

  if id_queue is not None:
    # If a worker dies during init, the pool will just create a replacement.
    id_queue.put(worker_proc.ident, block=True, timeout=0.1)


def next_sample(uid):
  # [FUNCTION DOC]
  # HELPER: worker task for the generator enqueuer. Input: uid, integer identifying a shared
  # generator. Returns its next value; raises a key error for an unknown id and end-of-iteration
  # when the generator is exhausted.
  """Gets the next value from the generator `uid`.

  To allow multiple generators to be used at the same time, we use `uid` to
  get a specific one. A single generator would cause the validation to
  overwrite the training generator.

  Args:
      uid: int, generator identifier

  Returns:
      The next value of generator `uid`.
  """
  return next(_SHARED_SEQUENCES[uid])


class GeneratorEnqueuer(SequenceEnqueuer):
  # [CLASS DOC]
  # Enqueuer for a plain Python generator. Workers repeatedly fetch the next generator value.
  # A finite generator ends the stream gracefully. Extra state: random seed for workers.
  # With threads and several workers the generator must be thread-safe.
  """Builds a queue out of a data generator.

  The provided generator can be finite in which case the class will throw
  a `StopIteration` exception.

  Args:
      generator: a generator function which yields data
      use_multiprocessing: use multiprocessing if True, otherwise threading
      random_seed: Initial seed for workers,
          will be incremented by one for each worker.
  """

  def __init__(self, generator,
               use_multiprocessing=False,
               random_seed=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: generator, the data generator; use_multiprocessing, boolean (default
    # false); random_seed, optional base seed for workers. Calls the base constructor and stores
    # the seed.
    super(GeneratorEnqueuer, self).__init__(generator, use_multiprocessing)
    self.random_seed = random_seed

  def _get_executor_init(self, workers):
    # [FUNCTION DOC]
    # HELPER: like the ordered variant but passes this enqueuer's random seed to the worker
    # initializer. Input: workers. Returns a pool factory.
    """Gets the Pool initializer for multiprocessing.

    Args:
      workers: Number of works.

    Returns:
        A Function to initialize the pool
    """
    def pool_fn(seqs):
      # [FUNCTION DOC]
      # HELPER: factory returned above. Input: seqs, shared mapping. Creates a worker pool
      # initialized with it and the seed, registers it in the module-wide pool set, returns it.
      pool = get_pool_class(True)(
          workers, initializer=init_pool_generator,
          initargs=(seqs, self.random_seed, get_worker_id_queue()))
      _DATA_POOLS.add(pool)
      return pool
    return pool_fn

  def _run(self):
    # [FUNCTION DOC]
    # HELPER: background loop. Publishes the generator, creates a pool and keeps submitting
    # asynchronous next-sample tasks to the bounded queue (blocking when full) until the stop
    # event is set.
    """Submits request to the executor and queue the `Future` objects."""
    self._send_sequence()  # Share the initial generator
    with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
      while True:
        if self.stop_signal.is_set():
          return

        self.queue.put(
            executor.apply_async(next_sample, (self.uid,)), block=True)

  def get(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the thread-safety RuntimeError is raised without chaining and only when the
    # message contains the executing-generator text; other errors are re-raised after stopping.
    # Generator yielding produced values (non-nothing) while running. When the generator finishes
    # (end of iteration surfaces), drains the queue, waits for pending futures and yields the
    # successful non-nothing results. On other errors it stops the enqueuer and raises a runtime
    # error if the generator is not thread-safe, else re-raises the original error.
    """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.

    Yields:
        The next element in the queue, i.e. a tuple
        `(inputs, targets)` or
        `(inputs, targets, sample_weights)`.
    """
    try:
      while self.is_running():
        inputs = self.queue.get(block=True).get()
        self.queue.task_done()
        if inputs is not None:
          yield inputs
    except StopIteration:
      # Special case for finite generators
      last_ones = []
      while self.queue.qsize() > 0:
        last_ones.append(self.queue.get(block=True))
      # Wait for them to complete
      for f in last_ones:
        f.wait()
      # Keep the good ones
      last_ones = [future.get() for future in last_ones if future.successful()]
      for inputs in last_ones:
        if inputs is not None:
          yield inputs
    except Exception as e:  # pylint: disable=broad-except
      self.stop()
      if 'generator already executing' in str(e):
        raise RuntimeError(
            'Your generator is NOT thread-safe. '
            'Keras requires a thread-safe generator when '
            '`use_multiprocessing=False, workers > 1`. ')
      raise e
